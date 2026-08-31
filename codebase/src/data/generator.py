"""Synthetic test-case dataset with pass/fail labels correlated to change context."""

from __future__ import annotations
import random
from datetime import datetime, timedelta
from typing import List
import numpy as np
import pandas as pd

MODULES = [
    "auth-service", "payment-service", "order-service", "inventory-service",
    "user-service", "api-gateway", "notification-service", "report-service",
]

# Tests that tend to fail when certain change keywords appear
RISKY_TESTS = {
    "test_payment_refund_flow": ["refund", "timeout", "payment api", "retry"],
    "test_auth_token_refresh": ["jwt", "token", "auth", "session"],
    "test_order_concurrency": ["lock", "concurrent", "optimistic", "race"],
    "test_inventory_stock_update": ["inventory", "stock", "quantity", "warehouse"],
    "test_user_registration_email": ["email", "smtp", "notification", "template"],
    "test_checkout_cart_total": ["cart", "discount", "tax", "pricing"],
    "test_api_rate_limiting": ["rate limit", "throttle", "gateway", "quota"],
    "test_db_migration_compat": ["migration", "schema", "flyway", "liquibase"],
    "test_file_upload_virus_scan": ["upload", "virus", "scan", "multipart"],
    "test_report_export_large": ["export", "report", "oom", "memory", "batch"],
}

STABLE_TESTS = [
    "test_health_endpoint", "test_config_loading", "test_logging_format",
    "test_metrics_endpoint", "test_cors_headers", "test_static_asset_serve",
    "test_version_endpoint", "test_readiness_probe",
]

FAIL_COMMITS = [
    "Refactor refund API timeout handling",
    "Fix JWT expiration edge case",
    "Add optimistic locking to order service",
    "Change inventory stock calculation formula",
    "Update email template engine",
    "Adjust rate limit thresholds on gateway",
    "Major schema migration for user tables",
    "Increase file upload size limit",
    "Optimize large report export memory usage",
    "Rewrite payment retry logic",
]

PASS_COMMITS = [
    "Update README and docs",
    "Bump dependency patch version",
    "Add unit test for helper function",
    "Fix typo in log message",
    "Improve health check description",
    "Cosmetic UI change in admin panel",
    "Add missing license header",
]


def generate_test_dataset(n_samples: int = 5000, fail_ratio: float = 0.28, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    records = []
    risky_names = list(RISKY_TESTS.keys())

    for i in range(n_samples):
        will_fail = random.random() < fail_ratio
        module = random.choice(MODULES)

        if will_fail:
            test_name = random.choice(risky_names)
            commit = random.choice(FAIL_COMMITS)
            # Align commit somewhat with test risk keywords
            keywords = RISKY_TESTS[test_name]
            if random.random() < 0.5:
                commit = commit + " related to " + random.choice(keywords)
            hist_fail_rate = round(random.uniform(0.25, 0.75), 2)
            changed = f"{module.replace('-', '_')}.java, {random.choice(['Controller', 'Service', 'Repository'])}.java"
        else:
            if random.random() < 0.6:
                test_name = random.choice(STABLE_TESTS)
                commit = random.choice(PASS_COMMITS)
                hist_fail_rate = round(random.uniform(0.0, 0.12), 2)
            else:
                test_name = random.choice(risky_names)
                commit = random.choice(PASS_COMMITS)
                hist_fail_rate = round(random.uniform(0.05, 0.30), 2)
            changed = f"{random.choice(['README.md', 'docs/guide.md', 'pom.xml', 'build.gradle'])}"

        full_text = (
            f"Test: {test_name}\nModule: {module}\n"
            f"Commit: {commit}\nChanged: {changed}\n"
            f"HistoricalFailRate: {hist_fail_rate}"
        )
        records.append({
            "test_id": f"T-{i+1:06d}",
            "test_name": test_name,
            "module": module,
            "commit_message": commit,
            "changed_files": changed,
            "historical_fail_rate": hist_fail_rate,
            "full_text": full_text,
            "outcome": "fail" if will_fail else "pass",
            "label": 1 if will_fail else 0,
            "timestamp": datetime.now() - timedelta(hours=random.randint(1, 2000)),
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_test_dataset(300)
    print(df["outcome"].value_counts())
    print(df.head(2))
