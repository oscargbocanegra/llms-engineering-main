"""Validation helpers for Synthetic Data Studio."""

from __future__ import annotations

import pandas as pd

from llm_portfolio.synthetic_data_studio.schemas import DATASET_SCHEMAS


def basic_quality_checks(df: pd.DataFrame, schema_name: str) -> dict:
    schema = DATASET_SCHEMAS[schema_name]
    expected_columns = [column["name"] for column in schema["columns"]]
    return {
        "missing_columns": [column for column in expected_columns if column not in df.columns],
        "extra_columns": [column for column in df.columns if column not in expected_columns],
        "n_rows": len(df),
        "n_cols": df.shape[1],
    }


def build_quality_summary(checks: dict) -> str:
    return (
        f"Rows generated: {checks['n_rows']}\n"
        f"Extra columns: {checks['extra_columns']}\n"
        f"Missing columns: {checks['missing_columns']}\n"
    )
