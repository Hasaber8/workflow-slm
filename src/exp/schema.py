from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

Service = Literal[
    "checkout-api", "payments-svc", "auth-gateway", "web-frontend", "cart-svc", "search-api",
    "inventory-svc", "user-profile", "orders-db-proxy", "recommendation-svc",
    "notification-svc", "shipping-svc"
]
Region = Literal["us-east", "us-west", "eu-west", "ap-south"]
Symptom = Literal[
    "high_error_rate", "high_latency", "high_cpu", "high_memory",
    "db_pool_saturation", "traffic_surge", "service_down"
]
Metric = Literal["error_rate", "latency_p99", "cpu", "memory", "request_rate", "db_pool_utilization"]
RootCause = Literal[
    "bad_deploy", "config_change_error", "upstream_dependency_failure", "resource_exhaustion",
    "db_pool_exhausted", "traffic_spike", "credential_failure", "transient_noise", "unknown"
]
Runbook = Literal["RB-001", "RB-101", "RB-102", "RB-201", "RB-301", "RB-302", "RB-303", "RB-401", "RB-900"]
LogSig = Literal[
    "unhandled_exception", "upstream_timeout", "connection_refused", "oom_killed",
    "cpu_throttled", "db_pool_exhausted", "auth_failure", "config_parse_error", "none"
]


class ExtractAlert(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: Literal["extract_alert"]
    service: Optional[Service] = None
    symptom: Optional[Symptom] = None
    region: Optional[Region] = None
    started_at: Optional[str] = None            # absolute UTC, YYYY-MM-DDTHH:MM:SSZ, only if alert states one
    started_minutes_ago: Optional[int] = None   # only if alert states a relative time


class GetMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: Literal["get_metrics"]
    service: Service
    metric: Metric
    window_minutes: Literal[30]


class GetRecentDeploys(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: Literal["get_recent_deploys"]
    service: Service
    window_minutes: Literal[120]


class QueryLogs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: Literal["query_logs"]
    service: Service
    window_minutes: Literal[30]


class GetDependencies(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: Literal["get_dependencies"]
    service: Service


class CheckDependencyHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: Literal["check_dependency_health"]
    dependency: Service


class Conclude(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: Literal["conclude"]
    root_cause: RootCause
    runbook_id: Runbook
    culprit: Optional[str] = None               # deploy id or dependency name, else null
    log_signature: Optional[LogSig] = None      # signature recognized in logs, null if logs never read


class DraftSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: Literal["draft_summary"]
    text: str = Field(max_length=700)


Step = Annotated[
    Union[
        ExtractAlert, GetMetrics, GetRecentDeploys, QueryLogs, GetDependencies,
        CheckDependencyHealth, Conclude, DraftSummary
    ],
    Field(discriminator="step")
]


class UnifiedStep(BaseModel):
    """Flattened schema fallback for structured output backends that do not support oneOf / non-string literals."""
    step: Literal[
        "extract_alert", "get_metrics", "get_recent_deploys", "query_logs",
        "get_dependencies", "check_dependency_health", "conclude", "draft_summary"
    ]
    service: Optional[Service] = None
    symptom: Optional[Symptom] = None
    region: Optional[Region] = None
    started_at: Optional[str] = None
    started_minutes_ago: Optional[int] = None
    metric: Optional[Metric] = None
    window_minutes: Optional[int] = None
    dependency: Optional[Service] = None
    root_cause: Optional[RootCause] = None
    runbook_id: Optional[Runbook] = None
    culprit: Optional[str] = None
    log_signature: Optional[LogSig] = None
    text: Optional[str] = None


STEP_ADAPTER = TypeAdapter(Step)
STEP_JSON_SCHEMA = STEP_ADAPTER.json_schema()
FLATTENED_STEP_JSON_SCHEMA = TypeAdapter(UnifiedStep).json_schema()
