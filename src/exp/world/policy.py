from exp.world.services import RUNBOOKS, SERVICES

SEV_TABLE = {
    (1, "critical"): "sev1", (1, "elevated"): "sev2",
    (2, "critical"): "sev2", (2, "elevated"): "sev3",
    (3, "critical"): "sev3", (3, "elevated"): "sev4"
}

ACTIONS = {
    "sev1": "page_oncall_immediately",
    "sev2": "page_oncall",
    "sev3": "create_ticket",
    "sev4": "log_only"
}


def triage_decision(service: str | None, primary_status: str | None, root_cause: str, culprit: str | None):
    title = RUNBOOKS[root_cause][1]
    if service is None:
        return {"severity": "sev3", "action": "create_ticket", "team": "sre-oncall", "runbook_title": title}
    tier = SERVICES[service]["tier"]
    status = primary_status if primary_status in ("elevated", "critical") else "elevated"
    sev = "sev4" if root_cause == "transient_noise" else SEV_TABLE[(tier, status)]
    if root_cause == "unknown":
        team = "sre-oncall"
    elif root_cause == "upstream_dependency_failure" and culprit in SERVICES:
        team = SERVICES[culprit]["team"]
    else:
        team = SERVICES[service]["team"]
    return {"severity": sev, "action": ACTIONS[sev], "team": team, "runbook_title": title}
