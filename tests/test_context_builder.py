from app.context.context_builder import BuildInput, build_context
from app.domain.types import ConversationMessage, RecalledMemory


def _msg(role: str, content: str, i: int) -> ConversationMessage:
    return ConversationMessage(
        id=f"m{i}", conversation_id="c1", role=role, content=content, created_at=i
    )


def test_build_context_never_cuts_system_or_message():
    system = "S" * 10_000
    result = build_context(
        BuildInput(
            system=system,
            summary=None,
            history=[],
            memories=[],
            message="pergunta do plantonista",
        ),
        budgets={"summary": 5, "history": 5, "memories": 5},
    )
    assert result.system == system
    assert result.message == "pergunta do plantonista"


def test_build_context_truncates_summary_to_budget():
    result = build_context(
        BuildInput(
            system="sys",
            summary="x" * 100,
            history=[],
            memories=[],
            message="oi",
        ),
        budgets={"summary": 10},
    )
    # budget=10 tokens -> max 40 chars
    assert len(result.summary) <= 40


def test_build_context_drops_oldest_history_first():
    history = [_msg("user", "m" * 50, i) for i in range(10)]
    result = build_context(
        BuildInput(system="s", summary=None, history=history, memories=[], message="oi"),
        budgets={"history": 20},
    )
    assert len(result.history) < len(history)
    # Sobreviventes devem ser os mais recentes (maiores created_at).
    assert result.history[-1].created_at == history[-1].created_at


def test_build_context_drops_lowest_score_memories_first():
    memories = [
        RecalledMemory(id="a", fact="fato A " * 5, score=0.9),
        RecalledMemory(id="b", fact="fato B " * 5, score=0.3),
        RecalledMemory(id="c", fact="fato C " * 5, score=0.6),
    ]
    result = build_context(
        BuildInput(system="s", summary=None, history=[], memories=memories, message="oi"),
        budgets={"memories": 10},
    )
    kept_ids = {m.id for m in result.memories}
    assert "b" not in kept_ids or len(result.memories) == len(memories)


def test_build_context_enriches_message_with_memories_and_summary():
    memories = [RecalledMemory(id="a", fact="prioriza checkout", score=0.9)]
    result = build_context(
        BuildInput(
            system="s",
            summary="resumo anterior",
            history=[],
            memories=memories,
            message="organize o plantão",
        ),
    )
    assert "Relevant memories:" in result.enriched_message
    assert "prioriza checkout" in result.enriched_message
    assert "Conversation summary:" in result.enriched_message
    assert "organize o plantão" in result.enriched_message
