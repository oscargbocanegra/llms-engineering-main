"""End-to-end orchestration helpers for Synthetic Data Studio."""

from __future__ import annotations

import tempfile
from collections.abc import Callable

import pandas as pd

from llm_portfolio.synthetic_data_studio.parsing import parse_csv_to_df
from llm_portfolio.synthetic_data_studio.prompting import build_prompt
from llm_portfolio.synthetic_data_studio.validation import (
    basic_quality_checks,
    build_quality_summary,
)

GeneratorFn = Callable[[str], str]


def run_synthetic_data_pipeline(
    generator: GeneratorFn,
    schema_name: str,
    n_rows: int,
    extra_instructions: str = "",
) -> tuple[str, pd.DataFrame, str]:
    prompt = build_prompt(schema_name, n_rows, extra_instructions)
    raw_output = generator(prompt)
    dataframe = parse_csv_to_df(raw_output)
    checks = basic_quality_checks(dataframe, schema_name)
    summary = build_quality_summary(checks)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    dataframe.to_csv(tmp_file.name, index=False)
    tmp_file_path = tmp_file.name
    tmp_file.close()

    return summary, dataframe, tmp_file_path
