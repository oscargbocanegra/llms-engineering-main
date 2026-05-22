"""Prompt construction helpers for Synthetic Data Studio."""

from llm_portfolio.synthetic_data_studio.schemas import DATASET_SCHEMAS


def build_prompt(schema_name: str, n_rows: int, extra_instructions: str = "") -> str:
    schema = DATASET_SCHEMAS[schema_name]
    lines: list[str] = []

    lines.append(
        "You are a synthetic tabular data generator for analytics and machine learning testing."
    )
    lines.append(
        "Your task is to generate a SYNTHETIC dataset in CSV format, without real personal data."
    )
    lines.append(f"Dataset: {schema_name}")
    lines.append(f"Description: {schema['description']}")
    lines.append("")
    lines.append("Column Specifications:")

    for column in schema["columns"]:
        lines.append(
            f"- {column['name']} ({column['type']}): {column['constraints']}"
        )

    lines.append("")
    lines.append(f"Generate exactly {n_rows} rows of data. You MUST produce {n_rows} rows.")
    lines.append("Do not write any text before or after the CSV. Only the CSV.")
    lines.append("Very important:")
    lines.append("1. Output MUST be in CSV format only.")
    lines.append("2. First row must be the header with column names.")
    lines.append("3. Do not include explanations, comments, or additional text.")
    lines.append("4. Respect data types and ranges as best as possible.")

    if extra_instructions:
        lines.append("")
        lines.append("Additional user instructions:")
        lines.append(extra_instructions)

    return "\n".join(lines)
