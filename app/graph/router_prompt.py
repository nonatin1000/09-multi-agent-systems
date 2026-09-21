"""Tabela de decisão + prompt de sistema do roteador do production-graph
(rotas: react | planExecute | reflect | team).
Equivalente a ``src/graph/router-prompt.ts``."""

from __future__ import annotations

ROUTER_DECISION_TABLE = """| Quando | Rota |
|--------|------|
| Consulta pontual / tool call simples (listar alertas, status, um serviço) | react |
| Pedido multi-passo / plano explícito / vários serviços ou etapas | planExecute |
| Pedido que exige verificação / alta criticidade / "revise" / resposta auditada | reflect |
| Investigação + plano + execução coordenadas; pedido complexo que se beneficia de papéis distintos (analista/planejador/executor) | team |"""

ROUTER_SYSTEM_PROMPT = "\n".join(
    [
        "Você é o roteador do OpsPilot. Escolha exatamente uma estratégia para o "
        "pedido do plantonista.",
        "Responda só com a estrutura { route, reason }.",
        "",
        "Tabela de decisão:",
        ROUTER_DECISION_TABLE,
        "",
        "reason: uma frase justificando a escolha.",
    ]
)
