import pytest
from exp.world.policy import triage_decision


def test_policy_all_tier_status_pairs():
    # Tier 1 (e.g. checkout-api)
    # (1, critical) -> sev1, page_oncall_immediately
    dec = triage_decision("checkout-api", "critical", "bad_deploy", "d-1")
    assert dec["severity"] == "sev1"
    assert dec["action"] == "page_oncall_immediately"
    assert dec["team"] == "team-checkout"

    # (1, elevated) -> sev2, page_oncall
    dec = triage_decision("checkout-api", "elevated", "bad_deploy", "d-1")
    assert dec["severity"] == "sev2"
    assert dec["action"] == "page_oncall"

    # Tier 2 (e.g. cart-svc)
    # (2, critical) -> sev2, page_oncall
    dec = triage_decision("cart-svc", "critical", "db_pool_exhausted", None)
    assert dec["severity"] == "sev2"
    assert dec["action"] == "page_oncall"
    assert dec["team"] == "team-checkout"

    # (2, elevated) -> sev3, create_ticket
    dec = triage_decision("cart-svc", "elevated", "db_pool_exhausted", None)
    assert dec["severity"] == "sev3"
    assert dec["action"] == "create_ticket"

    # Tier 3 (e.g. recommendation-svc)
    # (3, critical) -> sev3, create_ticket
    dec = triage_decision("recommendation-svc", "critical", "traffic_spike", None)
    assert dec["severity"] == "sev3"
    assert dec["action"] == "create_ticket"
    assert dec["team"] == "team-discovery"

    # (3, elevated) -> sev4, log_only
    dec = triage_decision("recommendation-svc", "elevated", "traffic_spike", None)
    assert dec["severity"] == "sev4"
    assert dec["action"] == "log_only"


def test_policy_transient_noise():
    # Transient noise is always sev4 / log_only regardless of tier
    dec = triage_decision("checkout-api", "critical", "transient_noise", None)
    assert dec["severity"] == "sev4"
    assert dec["action"] == "log_only"
    assert dec["runbook_title"] == "Close as noise"


def test_policy_unknown_cause():
    # Unknown cause with service -> team is sre-oncall
    dec = triage_decision("checkout-api", "critical", "unknown", None)
    assert dec["severity"] == "sev1"
    assert dec["team"] == "sre-oncall"

    # Unknown cause without service (service is None)
    dec_none = triage_decision(None, None, "unknown", None)
    assert dec_none["severity"] == "sev3"
    assert dec_none["action"] == "create_ticket"
    assert dec_none["team"] == "sre-oncall"


def test_policy_upstream_dependency_override():
    # When root cause is upstream_dependency_failure, team is culprit's team
    dec = triage_decision("checkout-api", "critical", "upstream_dependency_failure", "payments-svc")
    assert dec["team"] == "team-payments"

    # If culprit not in catalog, fallback to service team
    dec2 = triage_decision("checkout-api", "critical", "upstream_dependency_failure", "external-partner")
    assert dec2["team"] == "team-checkout"
