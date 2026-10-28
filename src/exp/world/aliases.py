"""Public domain vocabulary shared by alert generation and SOP rendering."""

SERVICE_ALIASES = {
    "checkout-api": ["checkout-api", "checkout service", "co-api", "checkout api", "chkout-svc"],
    "payments-svc": ["payments-svc", "payments service", "pay-svc", "payments api", "payment gateway"],
    "auth-gateway": ["auth-gateway", "auth gw", "authentication gateway", "identity proxy", "auth-gw"],
    "web-frontend": ["web-frontend", "frontend web", "storefront-ui", "web fe", "frontend"],
    "cart-svc": ["cart-svc", "shopping cart", "cart service", "basket-svc", "cart api"],
    "search-api": ["search-api", "search service", "discovery search", "search-svc"],
    "inventory-svc": ["inventory-svc", "stock service", "inventory api", "warehouse-inv"],
    "user-profile": ["user-profile", "user profile svc", "account service", "user-svc"],
    "orders-db-proxy": ["orders-db-proxy", "orders proxy", "db-proxy-orders", "orders database proxy"],
    "recommendation-svc": ["recommendation-svc", "recs service", "recommendations api", "recs-engine"],
    "notification-svc": ["notification-svc", "notify service", "comms-svc", "notifications"],
    "shipping-svc": ["shipping-svc", "shipping service", "fulfilment-shipping", "delivery-svc"],
}

REGION_ALIASES = {
    "us-east": ["us-east", "us-east-1", "virginia", "iad"],
    "us-west": ["us-west", "us-west-2", "oregon", "pdx"],
    "eu-west": ["eu-west", "eu-west-1", "dublin", "europe"],
    "ap-south": ["ap-south", "ap-south-1", "mumbai", "india"],
}
