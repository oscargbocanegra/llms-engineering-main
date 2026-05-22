"""CSV parsing helpers for Synthetic Data Studio."""

from __future__ import annotations

import io
import re

import pandas as pd


def parse_csv_to_df(text: str) -> pd.DataFrame:
    cleaned = re.sub(r"```(?:csv)?", "", text)
    cleaned = cleaned.strip("` \n")

    lines = [line for line in cleaned.splitlines() if "," in line]
    if not lines:
        return pd.DataFrame()

    csv_text = "\n".join(lines)
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return pd.DataFrame()
