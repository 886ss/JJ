"""
数据加载与预处理模块
功能：生成模拟电商数据 / 加载真实数据，完成清洗和特征工程
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_ecommerce_data(n_orders: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    生成模拟的电商交易数据，包含以下字段：
      - order_id: 订单 ID
      - customer_id: 客户 ID
      - product_category: 商品类别
      - price: 单价
      - quantity: 数量
      - order_date: 下单时间
      - state: 发货状态
      - payment_type: 支付方式
      - customer_state: 客户所在州

    参数:
        n_orders: 订单数量
        seed: 随机种子（保证可复现）

    返回:
        pd.DataFrame
    """
    np.random.seed(seed)

    # ---------- 商品类别池 ----------
    categories = [
        "电子产品", "服装", "家居用品", "美妆个护",
        "食品饮料", "图书", "运动户外", "母婴用品", "玩具"
    ]

    # ---------- 支付方式 ----------
    payment_types = ["信用卡", "借记卡", "支付宝", "微信支付", "货到付款"]

    # ---------- 发货状态 ----------
    states = ["已发货", "已签收", "待发货", "已退货"]

    # ---------- 巴西 27 个州（Olist 数据集风格） ----------
    brazilian_states = [
        "SP", "RJ", "MG", "RS", "PR", "SC", "BA", "PE", "CE",
        "PA", "MA", "GO", "PB", "RN", "ES", "AM", "MT", "DF",
        "MS", "SE", "RO", "TO", "AL", "PI", "AP", "AC", "RR"
    ]

    # ---------- 生成基础时间序列 ----------
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    days_range = (end_date - start_date).days

    # ---------- 生成客户 ID ----------
    n_customers = int(n_orders * 0.4)
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]

    # ---------- 构建订单数据 ----------
    order_dates = [start_date + timedelta(
        days=int(np.random.uniform(0, days_range))
    ) for _ in range(n_orders)]
    # 加随机小时数（0-23）
    order_dates = [d + timedelta(hours=int(np.random.uniform(0, 24))) for d in order_dates]
    order_dates.sort()

    data = {
        "order_id": [f"ORD_{i:06d}" for i in range(1, n_orders + 1)],
        "customer_id": np.random.choice(customer_ids, n_orders),
        "product_category": np.random.choice(categories, n_orders,
                                              p=[0.15, 0.15, 0.12, 0.10,
                                                 0.18, 0.05, 0.10, 0.08, 0.07]),
        "order_date": order_dates,
        "state": np.random.choice(states, n_orders, p=[0.40, 0.35, 0.15, 0.10]),
        "payment_type": np.random.choice(payment_types, n_orders,
                                         p=[0.30, 0.15, 0.25, 0.20, 0.10]),
        "customer_state": np.random.choice(brazilian_states, n_orders),
        "review_score": np.random.choice([1, 2, 3, 4, 5], n_orders,
                                         p=[0.08, 0.08, 0.12, 0.32, 0.40]),
    }

    df = pd.DataFrame(data)

    # ---------- 衍生价格（不同类型价格范围不同） ----------
    price_range_map = {
        "电子产品": (50, 3000), "服装": (30, 500), "家居用品": (20, 800),
        "美妆个护": (10, 300), "食品饮料": (5, 200), "图书": (10, 150),
        "运动户外": (20, 600), "母婴用品": (15, 400), "玩具": (10, 350),
    }
    df["price"] = df["product_category"].apply(
        lambda c: round(np.random.uniform(*price_range_map[c]), 2)
    )

    # ---------- 衍生数量 ----------
    df["quantity"] = np.random.choice([1, 2, 3, 4, 5], n_orders,
                                      p=[0.50, 0.25, 0.15, 0.07, 0.03])
    df["total_amount"] = (df["price"] * df["quantity"]).round(2)

    return df


def load_olist_data() -> pd.DataFrame:
    """
    从 Kaggle 加载 Olist 巴西电商真实数据集，转换为 Dashboard 所需格式。
    Olist 数据集: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
    """
    import os, kagglehub

    dataset_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    print(f"[OK] Olist 数据集路径: {dataset_path}")

    # 加载原始 CSV
    orders = pd.read_csv(os.path.join(dataset_path, "olist_orders_dataset.csv"))
    items = pd.read_csv(os.path.join(dataset_path, "olist_order_items_dataset.csv"))
    products = pd.read_csv(os.path.join(dataset_path, "olist_products_dataset.csv"))
    payments = pd.read_csv(os.path.join(dataset_path, "olist_order_payments_dataset.csv"))
    customers = pd.read_csv(os.path.join(dataset_path, "olist_customers_dataset.csv"))
    reviews = pd.read_csv(os.path.join(dataset_path, "olist_order_reviews_dataset.csv"))

    # 合并订单
    df = orders.merge(customers[["customer_id", "customer_state"]], on="customer_id", how="left")
    df = df.merge(items[["order_id", "product_id", "price", "freight_value"]], on="order_id", how="left")
    df = df.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")

    # 支付信息聚合（一个订单可能有多个支付方式，取金额最大的一种）
    payment_agg = (payments.groupby("order_id")
                   .agg(payment_type=("payment_type", "first"),
                        payment_value=("payment_value", "sum"))
                   .reset_index())
    df = df.merge(payment_agg, on="order_id", how="left")

    # 评分信息聚合（取每个订单第一条评分）
    review_agg = (reviews.groupby("order_id")
                  .agg(review_score=("review_score", "first"))
                  .reset_index())
    df = df.merge(review_agg, on="order_id", how="left")

    # 葡萄牙语品类名 → 中文
    category_cn = {
        "moveis_decoracao": "家居用品", "beleza_saude": "美妆个护", "esporte_lazer": "运动户外",
        "informatica_acessorios": "电子产品", "cama_mesa_banho": "家居用品", "brinquedos": "玩具",
        "ferramentas_jardim": "家居用品", "relogios_presentes": "服装", "telefonia": "电子产品",
        "automotivo": "运动户外", "eletrodomesticos": "电子产品", "pet_shop": "母婴用品",
        "livros_interesse_geral": "图书", "papelaria": "图书", "artes": "图书",
        "bebes": "母婴用品", "cool_stuff": "玩具", "eletronicos": "电子产品",
        "instrumentos_musicais": "运动户外", "alimentos": "食品饮料", "alimentos_bebidas": "食品饮料",
        "casa_conforto": "家居用品", "construcao_ferramentas": "家居用品",
        "fashion_bolsas_e_acessorios": "服装", "fashion_calcados": "服装",
        "fashion_esporte": "运动户外", "fashion_roupa_feminina": "服装",
        "fashion_roupa_infantil": "服装", "fashion_roupa_masculina": "服装",
        "fashion_underwear_e_moda_praia": "服装", "flores": "家居用品",
        "livros_importados": "图书", "livros_tecnicos": "图书", "malas_acessorios": "服装",
        "market_place": "食品饮料", "moveis_colchao": "家居用品", "moveis_escritorio": "家居用品",
        "moveis_sala": "家居用品", "musica": "图书", "portateis_casa_forno_e_cafe": "家居用品",
        "sinalizacao_e_seguranca": "家居用品", "tablets_impressao_imagem": "电子产品",
        "telefonia_fixa": "电子产品",
    }

    # 构建统一 DataFrame
    result = pd.DataFrame({
        "order_id": df["order_id"],
        "customer_id": df["customer_id"].astype(str),
        "product_category": df["product_category_name"].map(category_cn).fillna("其他"),
        "price": df["price"].fillna(0),
        "order_date": pd.to_datetime(df["order_purchase_timestamp"]),
        "state": df["order_status"].map({
            "delivered": "已签收", "shipped": "已发货", "invoiced": "待发货",
            "processing": "待发货", "approved": "待发货", "created": "待发货",
            "canceled": "已退货", "unavailable": "已退货",
        }).fillna("待发货"),
        "payment_type": df["payment_type"].map({
            "credit_card": "信用卡", "debit_card": "借记卡",
            "boleto": "货到付款", "voucher": "支付宝",
        }).fillna("其他支付"),
        "customer_state": df["customer_state"].fillna("SP"),
        "review_score": df["review_score"].fillna(3).astype(int).clip(1, 5),
    })

    # 衍生字段
    result["quantity"] = 1  # Olist 每行即一个商品项
    result["total_amount"] = result["price"] * result["quantity"]
    result = result.dropna(subset=["order_date"])
    result = result[result["price"] > 0]
    result = result.reset_index(drop=True)

    # 将日期整体平移 +7 年，从 2016-2018 变为 2023-2025
    result["order_date"] = result["order_date"] + pd.DateOffset(years=7)

    print(f"[OK] Olist 数据加载完成: {len(result):,} 条订单, "
          f"{result['customer_id'].nunique():,} 位客户, "
          f"日期范围 {result['order_date'].min().date()} ~ {result['order_date'].max().date()}")

    return result


def load_or_generate_data(n_orders: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    加载数据：优先 Olist 真实数据 → 本地 CSV 缓存 → 模拟生成
    """
    import os

    data_path = "data/ecommerce_data.csv"

    # 优先本地缓存
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=["order_date"])
        if len(df) > 5000:
            print(f"[OK] 从本地缓存加载: {data_path} ({len(df):,} 条)")
            return df

    # 尝试 Olist 真实数据
    try:
        df = load_olist_data()
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"[OK] Olist 数据已缓存至: {data_path}")
        return df
    except Exception as e:
        print(f"[提示] Olist 加载失败 ({e})，回退到模拟数据")

    # 回退：模拟数据
    print(f"[OK] 生成模拟数据 ({n_orders} 条订单)...")
    os.makedirs("data", exist_ok=True)
    df = generate_ecommerce_data(n_orders, seed)
    df.to_csv(data_path, index=False)
    print(f"[OK] 数据已保存至: {data_path}")
    return df


# ---------- 预处理流水线 ----------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗 + 特征工程
    """
    df = df.copy()

    # 解析日期
    df["order_date"] = pd.to_datetime(df["order_date"])

    # 衍生时间特征
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["order_weekday"] = df["order_date"].dt.day_name()
    df["order_hour"] = df["order_date"].dt.hour
    df["order_year_month"] = df["order_date"].dt.strftime("%Y-%m")

    # 衍生订单金额分档
    bins = [0, 50, 100, 200, 500, 1000, float("inf")]
    labels = ["0-50", "50-100", "100-200", "200-500", "500-1000", "1000+"]
    df["amount_tier"] = pd.cut(df["total_amount"], bins=bins, labels=labels)

    return df


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    计算核心 KPI
    """
    total_revenue = df["total_amount"].sum()
    total_orders = df["order_id"].nunique()
    total_customers = df["customer_id"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0

    # 环比上月增长
    max_date = df["order_date"].max()
    if pd.isna(max_date):
        return {"total_revenue": 0, "total_orders": 0, "total_customers": 0, "avg_order_value": 0, "revenue_mom": 0}
    this_month = max_date.strftime("%Y-%m")
    last_month = (max_date - pd.DateOffset(months=1)).strftime("%Y-%m")

    this_month_revenue = df[df["order_year_month"] == this_month]["total_amount"].sum()
    last_month_revenue = df[df["order_year_month"] == last_month]["total_amount"].sum()

    revenue_mom = ((this_month_revenue - last_month_revenue) / last_month_revenue
                   * 100) if last_month_revenue else 0

    return {
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "avg_order_value": avg_order_value,
        "revenue_mom": revenue_mom,
    }


def rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    RFM 客户分层分析
    R = Recency（距最近一次购买的天数）
    F = Frequency（购买次数）
    M = Monetary（消费总金额）
    """
    snapshot_date = df["order_date"].max() + timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        Recency=("order_date", lambda x: (snapshot_date - x.max()).days),
        Frequency=("order_id", "nunique"),
        Monetary=("total_amount", "sum"),
    ).reset_index()

    # 给每个指标打分（按四分位数分成 1-4 档）
    rfm["R_score"] = pd.qcut(rfm["Recency"].rank(method="first"), 4,
                             labels=[4, 3, 2, 1])
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 4,
                             labels=[1, 2, 3, 4])
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 4,
                             labels=[1, 2, 3, 4])

    rfm["RFM_score"] = (
        rfm["R_score"].astype(int)
        + rfm["F_score"].astype(int)
        + rfm["M_score"].astype(int)
    )

    # 客户分层
    def segment(score):
        if score >= 10:
            return "高价值客户"
        elif score >= 7:
            return "潜力客户"
        elif score >= 4:
            return "一般客户"
        else:
            return "流失风险客户"

    rfm["Segment"] = rfm["RFM_score"].apply(segment)

    return rfm
