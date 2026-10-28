import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional
from exp.runtime.clients import GeminiClient
from exp.world.aliases import REGION_ALIASES, SERVICE_ALIASES
from exp.world.services import REGIONS, SERVICES

PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "render_alert.md"

SYMPTOM_PHRASES = {
    "high_error_rate": ["5xx error rate elevated", "spike in 500 responses", "high error rate", "lots of failures", "error count surging"],
    "high_latency": ["p99 latency blown", "high latency across endpoints", "unusually slow response times", "sluggish performance", "latency spike"],
    "high_cpu": ["CPU utilization near 100%", "high cpu alert firing", "cpu exhaustion detected", "cpu load high"],
    "high_memory": ["memory usage exceeding threshold", "RAM saturation", "high memory usage alert", "memory pressure"],
    "db_pool_saturation": ["database connection pool exhausted", "db pool saturation", "db pool near max limit", "all db connections in use"],
    "traffic_surge": ["traffic spike detected", "massive request surge", "unexpected volume jump", "request rate 3x normal"],
    "service_down": ["service unresponsive", "health check failing", "service down", "outage detected"],
}


def introduce_typos(text: str, typo_level: int, rng: random.Random) -> str:
    if typo_level == 0 or len(text) < 10:
        return text
    chars = list(text)
    n_typos = min(typo_level, 4)
    for _ in range(n_typos):
        pos = rng.randint(0, len(chars) - 2)
        if chars[pos].isalpha() and chars[pos + 1].isalpha():
            # Swap adjacent characters
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def render_alert_templated(
    service: Optional[str],
    symptom: Optional[str],
    region: Optional[str],
    started_at: Optional[str],
    started_minutes_ago: Optional[int],
    received_at: str,
    style: str,
    typo_level: int = 1,
    rng: Optional[random.Random] = None
) -> str:
    """Renders a messy alert following style and recovery guidelines."""
    r = rng or random.Random()

    # Service representation
    if service is None:
        svc_str = r.choice(["billing-worker", "analytics-pipeline", "legacy-cron"])
    else:
        svc_str = r.choice(SERVICE_ALIASES.get(service, [service]))

    # Symptom representation
    if symptom is None:
        sym_str = "something seems off"
    else:
        sym_str = r.choice(SYMPTOM_PHRASES.get(symptom, ["unexpected anomaly"]))

    # Region representation
    reg_str = ""
    if region:
        reg_alias = r.choice(REGION_ALIASES.get(region, [region]))
        reg_str = f"in {reg_alias}"

    # Timing representation
    time_str = ""
    if started_minutes_ago is not None:
        time_str = f"started about {started_minutes_ago} min ago"
    elif started_at is not None:
        time_str = f"starting at {started_at}"

    # Render according to style
    if style == "pagerduty_style":
        core = f"[FIRING] {svc_str} {sym_str} {reg_str}, {time_str}. Check runbook dashboard."
    elif style == "slack_message":
        core = f"@channel heads up: {svc_str} is seeing {sym_str} {reg_str} ({time_str}). Investigating."
    elif style == "email_notification":
        core = f"Subject: Alert Notification: {svc_str}\nBody: Monitored metric alert: {sym_str} {reg_str}. Incident began {time_str}."
    elif style == "oncall_free_text":
        core = f"{svc_str} alert: {sym_str} {reg_str}, looks like {time_str}. Anyone else seeing this?"
    elif style == "terse_sms":
        core = f"ALERT: {svc_str} {sym_str} {reg_str} {time_str}".strip()
    elif style == "handoff_note":
        core = f"Handoff note: ongoing investigation into {svc_str} exhibiting {sym_str} {reg_str} ({time_str})."
    else:
        core = f"{svc_str} {sym_str} {reg_str} {time_str}"

    noisy_text = introduce_typos(core, typo_level, r)
    header = f"Received: {received_at}"
    return f"{header}\n{noisy_text}"


def render_alert_with_llm(
    client: GeminiClient,
    service: Optional[str],
    symptom: Optional[str],
    region: Optional[str],
    started_at: Optional[str],
    started_minutes_ago: Optional[int],
    received_at: str,
    style: str,
    typo_level: int = 1
) -> str:
    """Uses LLM to render alert via prompt."""
    prompt_tpl = PROMPT_FILE.read_text(encoding="utf-8")
    payload = {
        "service": service,
        "symptom": symptom,
        "region": region,
        "started_at": started_at,
        "started_minutes_ago": started_minutes_ago,
        "received_at": received_at,
        "style": style,
        "typo_level": typo_level
    }

    messages = [
        {"role": "system", "content": prompt_tpl},
        {"role": "user", "content": json.dumps(payload)}
    ]
    schema = {
        "type": "object",
        "properties": {"alert_text": {"type": "string"}},
        "required": ["alert_text"]
    }
    resp_text, _ = client.complete(messages, schema)
    try:
        data = json.loads(resp_text)
        alert_body = data.get("alert_text", "").strip()
    except Exception:
        alert_body = resp_text.strip()

    header = f"Received: {received_at}"
    return f"{header}\n{alert_body}"
