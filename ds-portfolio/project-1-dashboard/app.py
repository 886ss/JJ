"""
电商数据分析 Dashboard — 主应用

运行方式：
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="电商数据分析面板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- 加载本地模块 ----------
from data_loader import (
    load_or_generate_data,
    preprocess,
    compute_kpis,
    rfm_segmentation,
)
from visualizations import (
    sales_trend_line,
    category_bar,
    payment_pie,
    review_histogram,
    state_choropleth,
    rfm_segment_pie,
    weekday_heatmap,
    monthly_revenue_bar,
)

# ---------- 数据加载与缓存 ----------
@st.cache_data(ttl=60)
def get_data():
    """缓存数据加载，避免每次交互都重新计算"""
    df = load_or_generate_data(n_orders=12000)
    df = preprocess(df)
    rfm = rfm_segmentation(df)
    return df, rfm


df, rfm_df = get_data()

# ---------- 侧边栏 ----------
with st.sidebar:
    st.title("电商数据分析")

    st.markdown("---")

    # 日期筛选
    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()
    date_range = st.date_input(
        "选择日期范围",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # 品类多选筛选
    all_categories = sorted(df["product_category"].unique())
    selected_categories = st.multiselect(
        "商品品类",
        options=all_categories,
        default=all_categories,
    )

    # 支付方式筛选
    all_payments = sorted(df["payment_type"].unique())
    selected_payments = st.multiselect(
        "支付方式",
        options=all_payments,
        default=all_payments,
    )

    st.markdown("---")
    st.caption("*数据为模拟生成，仅供学习演示*")

# ---------- 数据筛选 ----------
mask = (
    (df["order_date"].dt.date >= date_range[0])
    & (df["order_date"].dt.date <= date_range[1])
    & (df["product_category"].isin(selected_categories))
    & (df["payment_type"].isin(selected_payments))
)
filtered_df = df[mask]

if len(filtered_df) == 0:
    st.error("当前筛选条件下无数据，请调整筛选条件")
    st.stop()

# 当筛选条件变化时重新计算 KPI
filtered_kpis = compute_kpis(filtered_df)

# ===================== 主页面 =====================
st.title("电商数据分析面板")
st.markdown("全链路销售数据监控：从概览到客户分层，一站式分析")
st.markdown("---")

# ---------- 第一行：KPI 卡片 ----------
st.subheader("核心指标")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "总销售额",
    f"R$ {filtered_kpis['total_revenue']:,.0f}",
    delta=f"{filtered_kpis['revenue_mom']:+.1f}% (环比)",
)
col2.metric(
    "订单数",
    f"{filtered_kpis['total_orders']:,}",
)
col3.metric(
    "客户数",
    f"{filtered_kpis['total_customers']:,}",
)
col4.metric(
    "客单价",
    f"R$ {filtered_kpis['avg_order_value']:,.2f}",
)
col5.metric(
    "退单率",
    f"{(filtered_df['state'] == '已退货').mean():.1%}",
    delta=f"{(filtered_df['state'] == '已退货').mean() - (df['state'] == '已退货').mean():.2%}",
    delta_color="inverse",
)

# ---------- 侧边栏动态统计 ----------
with st.sidebar:
    st.markdown("---")
    st.caption(f"当前筛选：{len(filtered_df):,} 订单 / {filtered_df['customer_id'].nunique():,} 客户")
    st.caption(f"数据范围：{date_range[0]} ~ {date_range[1]}")

st.markdown("---")

# ---------- 标签页：多维度分析 ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "销售趋势", "商品分析", "客户画像", "综合分析"
])

# ---- Tab 1: 销售趋势 ----
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.plotly_chart(sales_trend_line(filtered_df), use_container_width=True)

    with col_right:
        st.plotly_chart(monthly_revenue_bar(filtered_df), use_container_width=True)

    st.plotly_chart(weekday_heatmap(filtered_df), use_container_width=True)

# ---- Tab 2: 商品分析 ----
with tab2:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.plotly_chart(category_bar(filtered_df, top_n=10), use_container_width=True)

    with col_right:
        # 品类客单价对比
        cat_avg = (
            filtered_df.groupby("product_category")["price"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(
            cat_avg, x="price", y="product_category",
            orientation="h", color="price",
            color_continuous_scale="Greens",
            title="各品类平均单价",
        )
        fig.update_layout(
            xaxis_title="平均单价 (R$)",
            yaxis_title="",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 品类评分分布
    cat_review = (
        filtered_df.groupby(["product_category", "review_score"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        cat_review, x="product_category", y="count", color="review_score",
        title="各品类评分分布",
        color_continuous_scale="RdYlGn",
    )
    fig.update_layout(
        xaxis_title="品类", yaxis_title="订单数",
        height=400,
        margin=dict(l=20, r=20, t=50, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: 客户画像 ----
with tab3:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.plotly_chart(rfm_segment_pie(rfm_df), use_container_width=True)

    with col_right:
        st.plotly_chart(payment_pie(filtered_df), use_container_width=True)

    st.plotly_chart(state_choropleth(filtered_df), use_container_width=True)
    st.plotly_chart(review_histogram(filtered_df), use_container_width=True)

# ---- Tab 4: 综合分析 ----
with tab4:
    st.subheader("数据概览表")

    # 按品类 × 月份的交叉透视表
    pivot = filtered_df.pivot_table(
        values="total_amount",
        index="product_category",
        columns="order_year_month",
        aggfunc="sum",
        fill_value=0,
    )
    st.dataframe(
        pivot.style.background_gradient(cmap="Blues", axis=1)
        .format("R$ {:,.0f}"),
        use_container_width=True,
        height=400,
    )

    st.subheader("原始数据抽样")
    st.dataframe(
        filtered_df.head(200),
        use_container_width=True,
        hide_index=True,
    )

# ---------- 页脚 ----------
st.markdown("---")
st.caption("电商数据分析面板 · 数据科学项目集 · 仅供学习展示")
st.caption("技术栈：Python + Streamlit + Pandas + Plotly")
