"""Sumarização em lote do histórico de conversa
(equivalente a ``src/chat/history-summarizer.ts``)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.agents.model import OpsChatModel
from app.context.tokens import estimate_tokens
from app.domain.types import ConversationMessage, ConversationStore, TraceEvent

#: Máximo de mensagens anteriores injetadas cruas em um turno de estratégia.
HISTORY_LIMIT = 8

#: Mensagens por lote de sumarização (deve deixar a janela recente antes do gatilho).
SUMMARY_BATCH_SIZE = 8

#: Alvo suave de tamanho do resumo (estimativa chars/4).
SUMMARY_TOKEN_TARGET = 150

SUMMARIZER_PROMPT = (
    "Comprima o trecho de conversa a seguir em no máximo 150 tokens, preservando "
    "obrigatoriamente: decisões tomadas, fatos estabelecidos (nomes, datas, prazos, "
    "preferências), incidentes abertos/resolvidos e pendências abertas. Descarte "
    "cumprimentos e conversa social. Se houver um resumo anterior, incorpore-o. "
    "Responda só o resumo, em tópicos telegráficos"
)

ConversationSummarizer = Callable[
    [str | None, list[ConversationMessage]], Awaitable[str]
]


def format_summary_for_prompt(summary: str | None, body: str) -> str:
    trimmed = (summary or "").strip()
    if not trimmed:
        return body
    return f"Conversation summary:\n{trimmed}\n\n{body}"


def create_fake_conversation_summarizer(
    on_call: Callable[[str | None, list[ConversationMessage]], None] | None = None,
) -> ConversationSummarizer:
    """Fake determinístico para testes; trunca a ~SUMMARY_TOKEN_TARGET tokens."""

    async def _summarize(
        previous_summary: str | None, batch: list[ConversationMessage]
    ) -> str:
        if on_call:
            on_call(previous_summary, batch)
        prev = previous_summary or ""
        batch_part = ";".join(m.content for m in batch)
        raw = f"merge({prev})|batch:{batch_part}"
        max_chars = SUMMARY_TOKEN_TARGET * 4
        return raw if len(raw) <= max_chars else raw[:max_chars]

    return _summarize


def create_llm_conversation_summarizer(
    model_factory: Callable[[], OpsChatModel],
) -> ConversationSummarizer:
    async def _summarize(
        previous_summary: str | None, batch: list[ConversationMessage]
    ) -> str:
        batch_text = "\n".join(f"{m.role}: {m.content}" for m in batch)
        user_content = "\n\n".join(
            [
                f"Resumo anterior:\n{previous_summary.strip()}"
                if previous_summary and previous_summary.strip()
                else "Resumo anterior: (nenhum)",
                f"Trecho a comprimir:\n{batch_text}",
            ]
        )
        result = await model_factory().ainvoke(
            [("system", SUMMARIZER_PROMPT), ("user", user_content)]
        )
        content = result.content
        text = content if isinstance(content, str) else str(content)
        trimmed = text.strip()
        if not trimmed:
            raise ValueError("empty summarizer output")
        return trimmed

    return _summarize


@dataclass(slots=True)
class SummarizeOutcome:
    summary_text: str
    event: TraceEvent


async def maybe_summarize(
    conversations: ConversationStore,
    conversation_id: str,
    summarizer: ConversationSummarizer,
) -> SummarizeOutcome | None:
    """Se >= SUMMARY_BATCH_SIZE mensagens saíram da janela recente desde a marca
    d'água, sumariza o próximo lote contíguo (no máximo um lote por chamada).
    Fail-safe: nunca lança."""
    try:
        total = conversations.count_messages(conversation_id)
        existing = conversations.get_summary(conversation_id)
        covered = existing.covered_count if existing else 0
        outside = max(0, total - HISTORY_LIMIT)
        pending = outside - covered

        if pending < SUMMARY_BATCH_SIZE:
            return None

        batch = conversations.messages_ascending(
            conversation_id, covered, SUMMARY_BATCH_SIZE
        )
        if len(batch) < SUMMARY_BATCH_SIZE:
            return None

        summary_text = (
            await summarizer(existing.text if existing else None, batch)
        ).strip()
        if not summary_text:
            return None

        # Força suavemente o alvo para saídas grandes demais (esp. LLM).
        max_chars = SUMMARY_TOKEN_TARGET * 4
        stored = (
            summary_text
            if estimate_tokens(summary_text) <= SUMMARY_TOKEN_TARGET
            else summary_text[:max_chars]
        )

        next_covered = covered + SUMMARY_BATCH_SIZE
        conversations.upsert_summary(conversation_id, stored, next_covered)

        return SummarizeOutcome(
            summary_text=stored,
            event=TraceEvent(type="summarize", content=stored, node="contexto"),
        )
    except Exception:  # noqa: BLE001
        return None
