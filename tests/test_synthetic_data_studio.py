from llm_portfolio.synthetic_data_studio import (
    DATASET_SCHEMAS,
    basic_quality_checks,
    build_prompt,
    parse_csv_to_df,
)


def test_build_prompt_includes_schema_name_and_row_count():
    prompt = build_prompt("Retail Sales", 3, "Generate low fraud rate")
    assert "Retail Sales" in prompt
    assert "exactly 3 rows" in prompt
    assert "Generate low fraud rate" in prompt


def test_parse_csv_to_df_parses_simple_csv():
    df = parse_csv_to_df("a,b\n1,2\n3,4")
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_basic_quality_checks_reports_expected_columns():
    df = parse_csv_to_df(
        "order_id,order_date,customer_id,country,product_category,unit_price,quantity,total_amount,is_fraud\n"
        "ORD-0001,2024-01-02,CUST-0001,Colombia,Electronics,99.9,1,99.9,False"
    )
    checks = basic_quality_checks(df, "Retail Sales")
    assert checks["missing_columns"] == []
    assert checks["extra_columns"] == []
    assert checks["n_rows"] == 1


def test_schema_registry_contains_retail_sales():
    assert "Retail Sales" in DATASET_SCHEMAS
