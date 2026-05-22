"""Schema definitions extracted from Project 3: Synthetic Data Studio."""

DATASET_SCHEMAS = {
    "Retail Sales": {
        "description": "E-commerce retail sales transactions with fraud detection.",
        "columns": [
            {"name": "order_id", "type": "string", "constraints": "unique, format ORD-XXXX"},
            {"name": "order_date", "type": "date", "constraints": "between 2024-01-01 and 2024-12-31"},
            {"name": "customer_id", "type": "string", "constraints": "format CUST-XXXX"},
            {"name": "country", "type": "category", "constraints": "Colombia, Mexico, Chile, Peru"},
            {"name": "product_category", "type": "category", "constraints": "Electronics, Clothing, Home"},
            {"name": "unit_price", "type": "float", "constraints": "between 5 and 200"},
            {"name": "quantity", "type": "int", "constraints": "between 1 and 10"},
            {"name": "total_amount", "type": "float", "constraints": "unit_price * quantity"},
            {"name": "is_fraud", "type": "bool", "constraints": "True if transaction is fraudulent, False otherwise"},
        ],
    },
    "Bank Transactions": {
        "description": "Banking transactions for savings accounts.",
        "columns": [
            {"name": "transaction_id", "type": "string", "constraints": "unique"},
            {"name": "customer_id", "type": "string", "constraints": "format CUST-XXXX"},
            {"name": "transaction_date", "type": "date", "constraints": "2024-01-01 to 2024-12-31"},
            {"name": "transaction_type", "type": "category", "constraints": "deposit, withdrawal, transfer"},
            {"name": "amount", "type": "float", "constraints": "between 10 and 5000"},
            {"name": "balance_after", "type": "float", "constraints": "coherent balance after transaction"},
            {"name": "channel", "type": "category", "constraints": "branch, mobile_app, web, ATM"},
            {"name": "is_flagged", "type": "bool", "constraints": "True if suspicious, False otherwise"},
        ],
    },
    "Customer Support": {
        "description": "Customer support tickets for a SaaS business.",
        "columns": [
            {"name": "ticket_id", "type": "string", "constraints": "unique, format TCK-XXXX"},
            {"name": "created_at", "type": "datetime", "constraints": "within business hours"},
            {"name": "customer_segment", "type": "category", "constraints": "SMB, Mid-Market, Enterprise"},
            {"name": "issue_type", "type": "category", "constraints": "billing, login, bug, feature_request"},
            {"name": "priority", "type": "category", "constraints": "low, medium, high, critical"},
            {"name": "resolution_time_hours", "type": "float", "constraints": "between 0.5 and 72"},
            {"name": "csat_score", "type": "int", "constraints": "between 1 and 5"},
            {"name": "is_escalated", "type": "bool", "constraints": "True if escalated, False otherwise"},
        ],
    },
}
