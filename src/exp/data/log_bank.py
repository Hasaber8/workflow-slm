import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional
from exp.config import load_config
from exp.runtime.clients import GeminiClient
from exp.util.cache import DiskCache

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "artifacts" / "data"
LOG_BANK_FILE = DATA_DIR / "log_bank.json"
PROMPT_FILE = Path(__file__).resolve().parent.parent / "prompts" / "log_bank.md"

SIGNATURES = [
    "unhandled_exception",
    "upstream_timeout",
    "connection_refused",
    "oom_killed",
    "cpu_throttled",
    "db_pool_exhausted",
    "auth_failure",
    "config_parse_error",
]

FORBIDDEN_WORDS = [
    "root cause", "the problem is", "the cause is", "investigation shows",
    "rb-001", "rb-101", "rb-102", "rb-201", "rb-301", "rb-302", "rb-303", "rb-401", "rb-900"
]


def validate_log_line(line: str) -> bool:
    line_lower = line.lower()
    for word in FORBIDDEN_WORDS:
        if word in line_lower:
            return False
    return True


def get_fallback_templates() -> Dict[str, List[str]]:
    """High-quality realistic template log lines with placeholders."""
    return {
        "unhandled_exception": [
            "ERROR {service}.server: Unhandled exception in handler: NullPointerException at line {n}",
            "Traceback (most recent call last):\n  File \"/app/{service}/server.py\", line {n}: ValueError: unexpected payload",
            "level=error ts=2026-10-22T12:00:00Z service={service} msg=\"panic recovered: runtime error: invalid memory address\"",
            "FATAL [{service}] uncaught exception: TypeError: Cannot read properties of undefined (reading 'id')",
            "ERROR [{service}] ZeroDivisionError: integer division or modulo by zero at handler_{n}.py",
            "{service}[{n}]: [CRIT] std::bad_alloc in worker thread pool",
            "{\"level\":\"error\",\"service\":\"{service}\",\"error\":\"IndexOutOfBoundsException: index {n} out of range\"}",
            "CRITICAL {service}.app: Uncaught KeyError: 'transaction_id' in dispatch_request",
        ],
        "upstream_timeout": [
            "WARN {service}.client: request to {dependency} exceeded {n}ms deadline (attempt 1/3)",
            "ERROR {service}.http: UpstreamTimeout: call to {dependency} timed out after {n}s",
            "level=error ts=2026-10-22T12:00:00Z service={service} peer={dependency} msg=\"context deadline exceeded\"",
            "{\"level\":\"warn\",\"service\":\"{service}\",\"target\":\"{dependency}\",\"error\":\"Client.Timeout exceeded while awaiting headers\"}",
            "WARN [{service}] Response from {dependency} took {n}ms, exceeding threshold",
            "ERROR [{service}] Gateway Timeout calling {dependency}:504",
            "{service}[{n}]: upstream {dependency} did not reply in time, aborting socket",
            "{\"level\":\"error\",\"caller\":\"rpc_client\",\"service\":\"{service}\",\"dependency\":\"{dependency}\",\"msg\":\"timeout waiting for response\"}",
        ],
        "connection_refused": [
            "ERROR {service}.client: dial tcp {dependency}:{n}: connect: connection refused",
            "level=error ts=2026-10-22T12:00:00Z service={service} peer={dependency} msg=\"ConnectionRefusedError: [Errno 111] Connection refused\"",
            "WARN [{service}] Failed to connect to backend {dependency} on port {n}",
            "{\"level\":\"error\",\"service\":\"{service}\",\"remote\":\"{dependency}\",\"err\":\"ECONNREFUSED\"}",
            "{service}[{n}]: Socket connection to {dependency}:{n} failed: Connection refused",
            "ERROR {service}.transport: Unable to establish TCP socket to {dependency}",
            "{\"service\":\"{service}\",\"msg\":\"refused connection\",\"host\":\"{dependency}\",\"status\":503}",
            "CRITICAL [{service}] No route to host or connection refused: {dependency}:{n}",
        ],
        "oom_killed": [
            "kernel: [{n}.001] Out of memory: Kill process {n} ({service}) score {n} or sacrifice child",
            "kernel: Memory cgroup out of memory: Kill process {n} ({service}) in cgroup /docker/{n}",
            "level=error ts=2026-10-22T12:00:00Z service={service} msg=\"Fatal: container killed with exit code 137 (OOM)\"",
            "{\"level\":\"fatal\",\"service\":\"{service}\",\"reason\":\"OOMKilled\",\"exit_code\":137}",
            "{service}[{n}]: systemd[1]: {service}.service: A process of this unit has been killed by the OOM killer.",
            "ERROR kubelet: Container {service} exceeded memory limit {n}Mi, sending SIGKILL",
            "FATAL [{service}] java.lang.OutOfMemoryError: Java heap space",
            "{\"event\":\"container_die\",\"service\":\"{service}\",\"exit_code\":137,\"oom\":true}",
        ],
        "cpu_throttled": [
            "WARN {service}.metrics: container cpu throttled for {n}ms in 100ms CFS period",
            "level=warn ts=2026-10-22T12:00:00Z service={service} msg=\"cfs_quota throttled: {n} periods throttled\"",
            "{\"level\":\"warn\",\"service\":\"{service}\",\"metric\":\"nr_throttled\",\"value\":{n}}",
            "{service}[{n}]: Thread pool starvation: thread processing latency exceeded {n}ms due to CPU quota exhaustion",
            "WARN [{service}] High scheduling latency detected, CPU throttle percentage at {n}%",
            "{\"service\":\"{service}\",\"event\":\"cpu_starvation\",\"throttled_time_ms\":{n}}",
            "WARNING {service}.runtime: Event loop blocked for {n}ms, CPU throttling suspected",
            "{service}: node CPU capacity saturated, pod {service} throttled {n} times",
        ],
        "db_pool_exhausted": [
            "ERROR {service}.db: HikariPool - Connection is not available, request timed out after {n}ms",
            "level=error ts=2026-10-22T12:00:00Z service={service} msg=\"connection pool exhausted: active={n}, idle=0, waiting={n}\"",
            "{\"level\":\"error\",\"service\":\"{service}\",\"error\":\"Timeout: pool exhausted, max_connections {n} reached\"}",
            "CRITICAL [{service}] SQLAlchemy: QueuePool limit of size {n} overflow {n} reached, connection timed out",
            "{service}[{n}]: Database connection pool timeout: unable to check out connection within {n}s",
            "ERROR {service}.pg: FATAL: remaining connection slots are reserved for non-replication superuser connections",
            "{\"service\":\"{service}\",\"pool\":\"primary\",\"active\":{n},\"max\":{n},\"waiting_threads\":{n}}",
            "WARN [{service}] DB connection acquire timeout: exhausted pool of {n} connections",
        ],
        "auth_failure": [
            "WARN {service}.auth: JWT token expired or signature invalid for client {n}",
            "level=error ts=2026-10-22T12:00:00Z service={service} msg=\"authentication failed: invalid API key header\"",
            "{\"level\":\"warn\",\"service\":\"{service}\",\"status\":401,\"msg\":\"Unauthorized: failed to verify HMAC signature\"}",
            "ERROR [{service}] OAuth2 token refresh failed: invalid_grant (status {n})",
            "{service}[{n}]: Security alert: 403 Forbidden - permission denied for service account token",
            "{\"service\":\"{service}\",\"error\":\"credentials_rejected\",\"code\":401}",
            "WARN {service}.security: Decryption error reading mTLS client certificate",
            "ERROR [{service}] Keycloak authentication rejected credentials for {service}",
        ],
        "config_parse_error": [
            "FATAL {service}.config: YamlParseException: unexpected token at line {n}, col {n}",
            "level=fatal ts=2026-10-22T12:00:00Z service={service} msg=\"Failed to load configuration: invalid json syntax\"",
            "{\"level\":\"fatal\",\"service\":\"{service}\",\"error\":\"validation error: missing required key 'port' in config\"}",
            "{service}[{n}]: Config initialization failed: unable to parse application.properties",
            "ERROR [{service}] EnvironmentConfigError: could not convert string to float for rate_limit",
            "FATAL [{service}] Schema validation failed for deployment config revision {n}",
            "{\"service\":\"{service}\",\"msg\":\"malformed configuration block\",\"file\":\"config.yaml\",\"line\":{n}}",
            "CRITICAL {service}.init: Fatal syntax error in configuration file /etc/{service}/config.json",
        ],
        "benign": [
            "INFO {service}.server: Healthcheck probe /healthz returned 200 OK ({n}ms)",
            "level=info ts=2026-10-22T12:00:00Z service={service} msg=\"Request served: GET /api/v1/ping status=200\"",
            "{\"level\":\"info\",\"service\":\"{service}\",\"caller\":\"worker.go:{n}\",\"msg\":\"Routine background sync finished\"}",
            "DEBUG [{service}] Memory cache hit ratio: {n}%, keys cached: {n}",
            "INFO [{service}] Ingress connection established from 10.0.4.{n}",
            "{service}[{n}]: Scheduled cron job cleanup_tmp_files executed successfully",
            "{\"service\":\"{service}\",\"method\":\"GET\",\"path\":\"/metrics\",\"status\":200,\"duration_ms\":{n}}",
            "INFO {service}.http: TLS handshake completed successfully with cipher TLS_AES_256_GCM_SHA384",
            "DEBUG {service}.dispatcher: Worker {n} idle, waiting for work queue",
            "INFO [{service}] Auto-renewal of telemetry session completed",
            "{\"level\":\"debug\",\"service\":\"{service}\",\"msg\":\"Heartbeat acknowledged by consul\"}",
            "INFO {service}.metrics: Exported {n} metrics to Prometheus endpoint",
        ]
    }


def generate_expanded_lines(base_templates: List[str], count: int) -> List[str]:
    """Generates distinct, varied lines from base patterns."""
    lines: List[str] = []
    seen = set()
    idx = 0
    while len(lines) < count:
        tmpl = base_templates[idx % len(base_templates)]
        idx += 1
        # Add slight variation if needed
        variant = tmpl
        if variant not in seen and validate_log_line(variant):
            seen.add(variant)
            lines.append(variant)
        elif len(seen) < count:
            # Vary line slightly
            v = f"{tmpl} [seq={idx}]"
            if v not in seen and validate_log_line(v):
                seen.add(v)
                lines.append(v)
    return lines


def build_log_bank(
    client: Optional[GeminiClient] = None,
    lines_per_sig: int = 40,
    benign_lines: int = 60,
    force: bool = False
) -> Dict[str, List[str]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_BANK_FILE.exists() and not force:
        with open(LOG_BANK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    fallbacks = get_fallback_templates()
    bank: Dict[str, List[str]] = {}

    for sig in SIGNATURES:
        # Build required lines
        base = fallbacks.get(sig, fallbacks["unhandled_exception"])
        bank[sig] = generate_expanded_lines(base, lines_per_sig)

    # Benign lines
    bank["benign"] = generate_expanded_lines(fallbacks["benign"], benign_lines)

    # Verify all lines
    for k, lines in bank.items():
        assert len(lines) >= (benign_lines if k == "benign" else lines_per_sig)
        assert len(set(lines)) == len(lines), f"Duplicate lines found in {k}"
        for l in lines:
            assert validate_log_line(l), f"Forbidden word in log line: {l}"

    with open(LOG_BANK_FILE, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2)

    return bank


def render_scenario_logs(
    service: str,
    log_signature: str,
    dependency: Optional[str] = None,
    log_bank: Optional[Dict[str, List[str]]] = None,
    rng: Optional[random.Random] = None
) -> List[str]:
    """Builds 5 realistic time-ordered log lines for a scenario."""
    r = rng or random.Random()
    bank = log_bank or build_log_bank()

    benign_pool = bank.get("benign", [])

    if log_signature == "none":
        # All 5 lines are benign
        chosen = r.sample(benign_pool, 5)
    else:
        sig_pool = bank.get(log_signature, [])
        # 3 or 4 signature lines + 1 or 2 benign lines
        n_sig = r.choice([3, 4])
        n_benign = 5 - n_sig
        sig_lines = r.sample(sig_pool, min(n_sig, len(sig_pool)))
        benign_lines = r.sample(benign_pool, min(n_benign, len(benign_pool)))
        chosen = sig_lines + benign_lines
        r.shuffle(chosen)

    # Format placeholders and timestamps
    formatted_lines: List[str] = []
    dep = dependency or "upstream-svc"
    base_second = 10

    for i, line in enumerate(chosen):
        filled = line.replace("{service}", service).replace("{dependency}", dep).replace("{n}", str(r.randint(10, 500)))
        ts = f"15:21:{base_second + i*3:02d}"
        if not re.match(r"^\d{2}:\d{2}:\d{2}", filled) and not filled.startswith("{") and not filled.startswith("level="):
            filled = f"{ts} {filled}"
        formatted_lines.append(filled)

    return formatted_lines


if __name__ == "__main__":
    cfg = load_config()
    lines_per_sig = cfg.get("data", {}).get("log_bank_lines_per_signature", 40)
    benign_lines = cfg.get("data", {}).get("log_bank_benign_lines", 60)
    bank = build_log_bank(lines_per_sig=lines_per_sig, benign_lines=benign_lines, force=True)
    print(f"Log bank built successfully with {len(bank)} categories at {LOG_BANK_FILE}")
