"""pytest for 电商数据分析 Dashboard"""
import pandas as pd
import numpy as np
from data_loader import (
    generate_ecommerce_data,
    preprocess,
    compute_kpis,
    rfm_segmentation,
)


class TestDataGeneration:
    def test_generates_correct_rows(self):
        df = generate_ecommerce_data(n_orders=500, seed=42)
        assert len(df) == 500

    def test_all_required_columns(self):
        df = generate_ecommerce_data(100, seed=1)
        expected = {"order_id", "customer_id", "product_category",
                    "order_date", "state", "payment_type", "customer_state",
                    "review_score", "price", "quantity", "total_amount"}
        assert set(df.columns) == expected

    def test_no_missing_values(self):
        df = generate_ecommerce_data(1000, seed=7)
        assert df.isnull().sum().sum() == 0

    def test_price_positive(self):
        df = generate_ecommerce_data(500, seed=3)
        assert (df["price"] > 0).all()

    def test_review_score_range(self):
        df = generate_ecommerce_data(500, seed=5)
        assert df["review_score"].between(1, 5).all()

    def test_deterministic_with_seed(self):
        df1 = generate_ecommerce_data(500, seed=42)
        df2 = generate_ecommerce_data(500, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestPreprocess:
    def test_adds_time_features(self):
        df = generate_ecommerce_data(500, seed=2)
        df2 = preprocess(df)
        for col in ["order_month", "order_weekday", "order_hour",
                    "order_year_month", "amount_tier"]:
            assert col in df2.columns

    def test_no_rows_lost(self):
        df = generate_ecommerce_data(500, seed=2)
        df2 = preprocess(df)
        assert len(df) == len(df2)


class TestKPIs:
    def test_returns_all_keys(self):
        df = generate_ecommerce_data(500, seed=2)
        df = preprocess(df)
        kpis = compute_kpis(df)
        for key in ["total_revenue", "total_orders", "total_customers",
                    "avg_order_value", "revenue_mom"]:
            assert key in kpis

    def test_revenue_positive(self):
        df = generate_ecommerce_data(500, seed=2)
        df = preprocess(df)
        kpis = compute_kpis(df)
        assert kpis["total_revenue"] > 0


class TestRFM:
    def test_all_users_segmented(self):
        df = generate_ecommerce_data(1000, seed=2)
        df = preprocess(df)
        rfm = rfm_segmentation(df)
        assert len(rfm) == df["customer_id"].nunique()

    def test_segment_labels(self):
        df = generate_ecommerce_data(1000, seed=2)
        df = preprocess(df)
        rfm = rfm_segmentation(df)
        valid = {"高价值客户", "潜力客户", "一般客户", "流失风险客户"}
        assert set(rfm["Segment"].unique()).issubset(valid)
