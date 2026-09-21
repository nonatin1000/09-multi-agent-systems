from app.domain.severity import SEVERITIES, is_severity, normalize_severity


def test_normalize_severity_maps_pager_aliases():
    assert normalize_severity("sev1") == "critical"
    assert normalize_severity("sev2") == "high"
    assert normalize_severity("sev3") == "medium"
    assert normalize_severity("sev4") == "low"


def test_normalize_severity_is_case_and_whitespace_insensitive():
    assert normalize_severity(" SEV1 ") == "critical"
    assert normalize_severity("Critical") == "critical"


def test_normalize_severity_passes_through_unknown_values():
    assert normalize_severity("sev5") == "sev5"
    assert normalize_severity(None) is None
    assert normalize_severity(42) == 42


def test_is_severity():
    for value in SEVERITIES:
        assert is_severity(value)
    assert not is_severity("sev1")
    assert not is_severity(1)
