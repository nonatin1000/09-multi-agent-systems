from langchain_core.messages import AIMessage, ToolMessage

from app.trace.builder import build_trace_from_messages


def test_build_trace_from_messages_thought_action_observation_answer():
    messages = [
        AIMessage(
            content="vou checar os alertas",
            tool_calls=[{"name": "list_alerts", "args": {"status": "firing"}, "id": "1"}],
        ),
        ToolMessage(content="Found 1 alert(s):\n- [alert-001] ...", tool_call_id="1"),
        AIMessage(content="**Resumo**\n1 alerta crítico."),
    ]
    trace = build_trace_from_messages(messages, node="react")

    types = [e.type for e in trace]
    assert types == ["thought", "action", "observation", "answer"]
    assert trace[1].tool == "list_alerts"
    assert trace[1].tool_args == {"status": "firing"}
    assert all(e.node == "react" for e in trace)


def test_build_trace_from_messages_falls_back_to_no_answer():
    trace = build_trace_from_messages([AIMessage(content="")], node="react")
    assert len(trace) == 1
    assert trace[0].type == "answer"
    assert trace[0].content == "No answer generated."
