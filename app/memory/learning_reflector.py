"""Destila aprendizados duráveis da mensagem do usuário para memória semântica
de longo prazo (equivalente a ``src/memory/learning-reflector.ts``)."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from app.agents.model import OpsChatModel
from app.domain.types import MemoryStore, RecalledMemory

#: Fato durável canônico para organização do plantão / listagem por severidade.
PLANTAO_ORG_MEMORY_FACT = (
    "Ao organizar o plantão ou listar incidentes/alertas nos Achados, ordenar por "
    "severidade: critical, depois high, medium, low."
)

LEARNING_REFLECTOR_PROMPT = " ".join(
    [
        "Você destila APRENDIZADOS DURÁVEIS da mensagem do usuário para memória de "
        "longo prazo de um copiloto de plantão (OpsPilot).",
        'hasLearning=true SOMENTE se a mensagem contiver preferência ou fato '
        'operacional DURÁVEL (ex.: "sempre priorize o serviço checkout", "prefiro '
        'resumos curtos").',
        "hasLearning=true também quando o usuário pedir para ORGANIZAR o plantão / "
        'listagem (ex.: "organize meu plantão", "organize e mostre os incidentes", '
        '"liste por prioridade/severidade"): grave um fato estável de preferência de '
        "ordenação — NÃO trate isso como pedido pontual descartável.",
        f'Nesse caso, fact pode ser: "{PLANTAO_ORG_MEMORY_FACT}" (ou equivalente em '
        "uma frase).",
        "hasLearning=false para pedidos PONTUAIS one-shot SEM organização/prioridade "
        '(ex.: "liste alertas", "abra um incidente", "resolva X agora", perguntas '
        "efêmeras de status).",
        "hasLearning=false se a mensagem contiver SEGREDOS ou dados sensíveis "
        "(senhas, tokens, API keys, credenciais, secrets).",
        "Quando hasLearning=true, fact deve ser um enunciado estável em uma frase "
        "(não copie o pedido pontual bruto).",
        "Quando hasLearning=false, fact deve ser string vazia.",
    ]
)


class LearningReflection(BaseModel):
    has_learning: bool = Field(
        description=(
            "true somente se a mensagem contiver preferência ou fato operacional "
            "DURÁVEL elegível a memória (nunca pedido pontual puro, nunca segredo; "
            "organizar plantão/listagem por prioridade CONTA como durável)"
        )
    )
    fact: str = Field(
        description=(
            "Enunciado estável em 1 frase; string vazia se has_learning=false. "
            "Nunca copie pedido pontual nem segredo."
        )
    )


LearningReflectorFn = Callable[[str], Awaitable[LearningReflection]]

_NO_LEARNING = LearningReflection(has_learning=False, fact="")

_ORGANIZES_RE = re.compile(r"\borganiz|\bpriorid|\bseveridade\b|\border", re.IGNORECASE)
_LISTING_CONTEXT_RE = re.compile(
    r"\bplant[aã]o\b|\bincident|\balerta|\blist|\bmostr", re.IGNORECASE
)


def suggests_plantao_organization(message: str) -> bool:
    """True quando a mensagem pede para organizar o plantão / listar por prioridade."""
    organizes = bool(_ORGANIZES_RE.search(message))
    listing_context = bool(_LISTING_CONTEXT_RE.search(message))
    return organizes and (listing_context or bool(re.search(r"\borganiz", message, re.IGNORECASE)))


def build_memory_recall_query(message: str) -> str:
    """Query de recall: quando o usuário pede para organizar o plantão, enriquece
    a query para que preferências duráveis de organização sejam recuperadas mesmo
    que a redação difira."""
    if not suggests_plantao_organization(message):
        return message
    return "\n".join(
        [
            message.strip(),
            "preferência organização plantão",
            "ordenar Achados por severidade critical high medium low",
        ]
    )


def create_llm_learning_reflector(
    model_factory: Callable[[], OpsChatModel],
) -> LearningReflectorFn:
    async def _reflect(user_message: str) -> LearningReflection:
        try:
            raw = (
                await model_factory()
                .with_structured_output(LearningReflection)
                .ainvoke(
                    [
                        ("system", LEARNING_REFLECTOR_PROMPT),
                        ("user", user_message),
                    ]
                )
            )
            return raw if isinstance(raw, LearningReflection) else LearningReflection(**raw)
        except Exception:  # noqa: BLE001
            return _NO_LEARNING

    return _reflect


async def schedule_learning(
    reflector: LearningReflectorFn,
    memories: MemoryStore,
    user_id: str,
    user_message: str,
) -> None:
    try:
        reflection = await reflector(user_message)
        if not reflection.has_learning:
            return
        fact = reflection.fact.strip()
        if not fact:
            return
        await memories.remember(user_id, fact)
    except Exception:  # noqa: BLE001
        # Best-effort: nunca quebra o turno do chat.
        pass


async def prepare_memories_for_turn(
    memories: MemoryStore,
    user_id: str,
    user_message: str,
    reflector: LearningReflectorFn | None = None,
) -> list[RecalledMemory]:
    """Persiste o aprendizado durável antes do recall para que o mesmo turno já
    possa usar o novo fato. Para pedidos de organização do plantão, aguarda o
    reflector; caso contrário, dispara e esquece (fire-and-forget)."""
    query = build_memory_recall_query(user_message)

    if reflector is None:
        return await memories.recall(user_id, query)

    if suggests_plantao_organization(user_message):
        await schedule_learning(reflector, memories, user_id, user_message)
    else:
        asyncio.ensure_future(
            schedule_learning(reflector, memories, user_id, user_message)
        )

    return await memories.recall(user_id, query)
