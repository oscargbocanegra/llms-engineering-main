"""Minimal smoke example for the extracted Synthetic Data Studio modules."""

from llm_portfolio.synthetic_data_studio import run_synthetic_data_pipeline


def fake_generator(_: str) -> str:
    return """order_id,order_date,customer_id,country,product_category,unit_price,quantity,total_amount,is_fraud\nORD-0001,2024-01-02,CUST-0001,Colombia,Electronics,99.9,1,99.9,False\nORD-0002,2024-01-03,CUST-0002,Peru,Home,45.0,2,90.0,False"""


if __name__ == "__main__":
    summary, dataframe, csv_path = run_synthetic_data_pipeline(
        generator=fake_generator,
        schema_name="Retail Sales",
        n_rows=2,
        extra_instructions="",
    )
    print(summary)
    print(dataframe.head())
    print(f"CSV written to: {csv_path}")
