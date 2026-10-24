SERVICES = {
    "checkout-api":       {"tier": 1, "team": "team-checkout",   "deps": ["auth-gateway", "payments-svc", "inventory-svc"]},
    "payments-svc":       {"tier": 1, "team": "team-payments",   "deps": ["auth-gateway", "orders-db-proxy"]},
    "auth-gateway":       {"tier": 1, "team": "team-identity",   "deps": ["user-profile"]},
    "web-frontend":       {"tier": 1, "team": "team-web",        "deps": ["checkout-api", "search-api", "cart-svc"]},
    "cart-svc":           {"tier": 2, "team": "team-checkout",   "deps": ["inventory-svc", "orders-db-proxy"]},
    "search-api":         {"tier": 2, "team": "team-discovery",  "deps": ["recommendation-svc"]},
    "inventory-svc":      {"tier": 2, "team": "team-fulfilment", "deps": ["orders-db-proxy"]},
    "user-profile":       {"tier": 2, "team": "team-identity",   "deps": []},
    "orders-db-proxy":    {"tier": 2, "team": "team-data",       "deps": []},
    "recommendation-svc": {"tier": 3, "team": "team-discovery",  "deps": []},
    "notification-svc":   {"tier": 3, "team": "team-comms",      "deps": ["user-profile"]},
    "shipping-svc":       {"tier": 3, "team": "team-fulfilment", "deps": ["inventory-svc", "notification-svc"]},
}

REGIONS = ["us-east", "us-west", "eu-west", "ap-south"]

RUNBOOKS = {
    "transient_noise": ("RB-001", "Close as noise"),
    "bad_deploy": ("RB-101", "Roll back the deploy"),
    "config_change_error": ("RB-102", "Revert the config change"),
    "upstream_dependency_failure": ("RB-201", "Escalate to the dependency owner"),
    "resource_exhaustion": ("RB-301", "Scale up capacity"),
    "db_pool_exhausted": ("RB-302", "Resize the connection pool"),
    "traffic_spike": ("RB-303", "Enable rate limiting"),
    "credential_failure": ("RB-401", "Rotate credentials"),
    "unknown": ("RB-900", "Escalate to a human"),
}

HELDOUT_SERVICES = ["shipping-svc", "notification-svc"]
TRAIN_SERVICES = [s for s in SERVICES if s not in HELDOUT_SERVICES]
