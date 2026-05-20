"""
可视化函数模块
提供 Dashboard 中所有图表的 Plotly 封装
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sales_trend_line(df: pd.DataFrame) -> go.Figure:
    """
    销售额与订单量日趋势图（双 Y 轴）
    """
    daily = (df.groupby(df["order_date"].dt.date)
             .agg(revenue=("total_amount", "sum"),
                  orders=("order_id", "nunique"))
             .reset_index())

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=daily["order_date"], y=daily["revenue"],
                   name="销售额", line=dict(color="#636EFA", width=2),
                   fill="tozeroy", fillcolor="rgba(99,110,250,0.1)"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=daily["order_date"], y=daily["orders"],
               name="订单数", marker_color="rgba(239,85,59,0.6)",
               marker_line_width=0),
        secondary_y=True,
    )

    fig.update_layout(
        title="日销售额与订单量趋势",
        hovermode="x unified",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_yaxes(title_text="销售额 (R$)", secondary_y=False)
    fig.update_yaxes(title_text="订单数", secondary_y=True)

    return fig


def category_bar(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    品类销售额排名（横向柱状图）
    """
    cat = (df.groupby("product_category")
           .agg(revenue=("total_amount", "sum"),
                avg_price=("price", "mean"))
           .sort_values("revenue", ascending=False)
           .head(top_n)
           .reset_index())

    fig = px.bar(cat, x="revenue", y="product_category",
                 orientation="h",
                 color="revenue",
                 color_continuous_scale="Blues",
                 text=cat["revenue"].apply(lambda x: f"R$ {x:,.0f}"))

    fig.update_traces(textposition="outside")
    fig.update_layout(
        title=f"品类销售额 TOP {top_n}",
        xaxis_title="销售额 (R$)",
        yaxis_title="",
        height=400,
        margin=dict(l=20, r=80, t=50, b=20),
        coloraxis_showscale=False,
    )

    return fig


def payment_pie(df: pd.DataFrame) -> go.Figure:
    """
    支付方式分布（环形图）
    """
    payment = (df.groupby("payment_type")
               .agg(count=("order_id", "nunique"))
               .reset_index())

    fig = px.pie(payment, names="payment_type", values="count",
                 hole=0.5,
                 color_discrete_sequence=px.colors.qualitative.Set2)

    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        title="支付方式分布",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def review_histogram(df: pd.DataFrame) -> go.Figure:
    """
    评分分布柱状图
    """
    review = df["review_score"].value_counts().sort_index().reset_index()
    review.columns = ["评分", "数量"]

    colors = ["#EF553B", "#FFA15A", "#FFD700", "#AB63FA", "#636EFA"]

    fig = px.bar(review, x="评分", y="数量",
                 color="评分",
                 color_discrete_map={i + 1: c for i, c in enumerate(colors)},
                 text="数量")

    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="评分分布",
        xaxis_title="评分",
        yaxis_title="订单数",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )

    return fig


def state_choropleth(df: pd.DataFrame) -> go.Figure:
    """
    按巴西各州销售额的地理热力图
    """
    state_data = (df.groupby("customer_state")
                  .agg(revenue=("total_amount", "sum"))
                  .reset_index())

    fig = go.Figure(data=go.Choropleth(
        locations=state_data["customer_state"],
        z=state_data["revenue"],
        locationmode="USA-states",  # 巴西用 ISO 3166-2
        # 注意：Plotly 的巴西地图需要 ISO 3166-2:BR-XX 格式
        # 这里做简化处理，用美国地图作为示意
        colorscale="Blues",
        colorbar=dict(title="销售额 (R$)"),
    ))

    # 实际上用巴西州数据，需要 geojson
    # 这里用柱状图替代作为更兼容的方案
    fig = px.bar(
        state_data.sort_values("revenue", ascending=False),
        x="customer_state", y="revenue",
        color="revenue",
        color_continuous_scale="Blues",
        text=state_data["revenue"].apply(lambda x: f"R$ {x/1000:,.0f}k")
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title="各州销售额分布",
        xaxis_title="州",
        yaxis_title="销售额 (R$)",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        coloraxis_showscale=False,
    )

    return fig


def rfm_segment_pie(rfm_df: pd.DataFrame) -> go.Figure:
    """
    RFM 客户分层环形图
    """
    seg = rfm_df["Segment"].value_counts().reset_index()
    seg.columns = ["客户分层", "人数"]

    colors_map = {
        "高价值客户": "#636EFA",
        "潜力客户": "#00CC96",
        "一般客户": "#AB63FA",
        "流失风险客户": "#EF553B",
    }

    fig = px.pie(seg, names="客户分层", values="人数",
                 hole=0.5,
                 color="客户分层",
                 color_discrete_map=colors_map)

    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        title="RFM 客户分层",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def weekday_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    按星期几 × 时段的订单分布热力图
    """
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
    hour_order = list(range(24))

    heat = (df.groupby(["order_weekday", "order_hour"])
            .agg(count=("order_id", "nunique"))
            .reset_index())

    pivot = heat.pivot(index="order_weekday", columns="order_hour",
                       values="count")
    pivot = pivot.reindex(index=weekday_order, columns=hour_order)
    pivot = pivot.fillna(0)

    # 中文星期映射
    cn_days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    fig = px.imshow(
        pivot,
        labels=dict(x="小时", y="星期", color="订单数"),
        x=[f"{h}:00" for h in hour_order],
        y=cn_days,
        color_continuous_scale="OrRd",
        aspect="auto",
    )

    fig.update_layout(
        title="星期 × 时段订单分布热力图",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def monthly_revenue_bar(df: pd.DataFrame) -> go.Figure:
    """
    月度销售额柱状图（含环比增长率）
    """
    monthly = (df.groupby("order_year_month")
               .agg(revenue=("total_amount", "sum"))
               .reset_index())
    monthly["增长率"] = monthly["revenue"].pct_change() * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=monthly["order_year_month"], y=monthly["revenue"],
               name="月销售额", marker_color="#636EFA",
               text=monthly["revenue"].apply(lambda x: f"R$ {x/1000:,.0f}k"),
               textposition="outside"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=monthly["order_year_month"], y=monthly["增长率"],
                   name="环比增长率", mode="lines+markers",
                   line=dict(color="#EF553B", width=2, dash="dot"),
                   marker=dict(size=6)),
        secondary_y=True,
    )

    fig.update_layout(
        title="月度销售额与环比增长率",
        hovermode="x unified",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_yaxes(title_text="销售额 (R$)", secondary_y=False)
    fig.update_yaxes(title_text="环比增长率 (%)", secondary_y=True)

    return fig
