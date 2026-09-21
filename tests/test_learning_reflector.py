from app.memory.learning_reflector import (
    build_memory_recall_query,
    suggests_plantao_organization,
)


def test_suggests_plantao_organization_true_cases():
    assert suggests_plantao_organization("organize meu plantão")
    assert suggests_plantao_organization("liste os incidentes por prioridade")
    assert suggests_plantao_organization("organize e mostre os alertas por severidade")


def test_suggests_plantao_organization_false_cases():
    assert not suggests_plantao_organization("liste alertas")
    assert not suggests_plantao_organization("abra um incidente para checkout")
    assert not suggests_plantao_organization("qual o status do github?")


def test_build_memory_recall_query_enriches_when_organizing():
    query = build_memory_recall_query("organize meu plantão")
    assert "organize meu plantão" in query
    assert "preferência organização plantão" in query


def test_build_memory_recall_query_passthrough_otherwise():
    assert build_memory_recall_query("liste alertas") == "liste alertas"
