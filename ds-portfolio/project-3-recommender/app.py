"""
电影推荐系统 — Streamlit 交互界面
运行: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_movielens_100k, load_latest_movies, get_hot_movies, get_recent_movies

st.set_page_config(
    page_title="电影推荐系统",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 加载数据与模型 ----
@st.cache_resource
def load_data_and_model():
    import joblib, os

    data = load_movielens_100k()

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    candidates = sorted([f for f in os.listdir(model_dir) if f.endswith(".pkl")], reverse=True)
    if not candidates:
        raise FileNotFoundError("No trained model found. Run: python train.py")
    bundle = joblib.load(os.path.join(model_dir, candidates[0]))
    return data, bundle["model"], bundle["config"]

with st.spinner("加载电影数据与推荐模型中…"):
    data, model, model_config = load_data_and_model()

movies_df = data["movies"]
ratings_df = data["ratings"]
n_users = data["n_users"]
n_items = data["n_items"]


@st.cache_data(ttl=3600)
def _load_latest():
    """延迟加载最新电影数据集。网络不可用时返回 None 并缓存结果，避免重复请求。"""
    try:
        return load_latest_movies()
    except Exception:
        return None


@st.cache_data(ttl=1800)
def get_cached_hot_movies():
    try:
        lm = _load_latest()
    except Exception:
        return []
    return get_hot_movies(lm, min_ratings=80, top_n=15) if lm is not None else []


@st.cache_data(ttl=1800)
def get_cached_recent_movies():
    try:
        lm = _load_latest()
    except Exception:
        return []
    return get_recent_movies(lm, min_ratings=50, top_n=15) if lm is not None else []

# 电影类型中文映射
GENRE_CN = {
    "Action": "动作", "Adventure": "冒险", "Animation": "动画", "Children's": "儿童",
    "Comedy": "喜剧", "Crime": "犯罪", "Documentary": "纪录", "Drama": "剧情",
    "Fantasy": "奇幻", "Film-Noir": "黑色电影", "Horror": "恐怖", "Musical": "音乐",
    "Mystery": "悬疑", "Romance": "爱情", "Sci-Fi": "科幻", "Thriller": "惊悚",
    "War": "战争", "Western": "西部",
}

# ---- 推荐函数 ----
def recommend_for_user(user_id, n=10):
    user_idx = user_id - 1
    if user_idx < 0 or user_idx >= n_users:
        return []
    interactions = data["interactions"]
    user_items = interactions[user_idx]
    recs = model.recommend(user_idx, user_items, N=n, filter_already_liked_items=True)
    item_indices, scores = recs[0], recs[1]
    result = []
    for idx, score in zip(item_indices, scores):
        movie = movies_df[movies_df["item_id"] == int(idx) + 1]
        if not movie.empty:
            row = movie.iloc[0]
            result.append({
                "item_id": int(idx) + 1,
                "title": row["clean_title"],
                "year": int(row["year"]) if not pd.isna(row.get("year")) else None,
                "genres": ", ".join(GENRE_CN.get(g, g) for g in row["genres"][:3]) if row["genres"] else "",
                "score": round(float(score), 4),
            })
    return result

def get_popular_movies(n=20):
    popular = (ratings_df.groupby("item_id")
               .agg(count=("rating", "count"), avg_rating=("rating", "mean"))
               .sort_values("count", ascending=False).head(n))
    movie_lookup = movies_df.set_index("item_id")
    result = []
    for item_id, row in popular.iterrows():
        if item_id not in movie_lookup.index:
            continue
        r = movie_lookup.loc[item_id]
        is_series = isinstance(r, pd.DataFrame)
        if is_series and r.empty:
            continue
        if is_series:
            r = r.iloc[0]
        result.append({
            "item_id": int(item_id),
            "title": r["clean_title"],
            "year": int(r["year"]) if not pd.isna(r.get("year")) else None,
            "genres": ", ".join(GENRE_CN.get(g, g) for g in r["genres"][:3]) if r["genres"] else "",
            "avg_rating": float(f"{row['avg_rating']:.1f}"),
            "num_ratings": int(row["count"]),
        })
    return result

def search_movies(query, limit=15):
    mask = movies_df["clean_title"].str.contains(query, case=False, na=False)
    results = movies_df[mask].head(limit)
    return [{
        "item_id": int(r["item_id"]),
        "title": r["clean_title"],
        "year": int(r["year"]) if not pd.isna(r["year"]) else None,
        "genres": ", ".join(GENRE_CN.get(g, g) for g in r["genres"][:3]) if r["genres"] else "",
    } for _, r in results.iterrows()]

def get_content_similar_items(item_id, n=10):
    """基于电影类型找到相似电影。"""
    if data.get("item_features") is None:
        return []
    item_feat = data["item_features"].toarray()
    item_idx = item_id - 1
    if item_idx < 0 or item_idx >= item_feat.shape[0]:
        return []
    sims = cosine_similarity(item_feat[item_idx:item_idx+1], item_feat)[0]
    top = np.argsort(-sims)[1:n+1]
    result = []
    for i in top:
        if sims[i] <= 0:
            continue
        movie = movies_df[movies_df["item_id"] == int(i) + 1]
        if not movie.empty:
            row = movie.iloc[0]
            result.append({
                "item_id": int(i) + 1,
                "title": row["clean_title"],
                "year": int(row["year"]) if not pd.isna(row.get("year")) else None,
                "genres": ", ".join(GENRE_CN.get(g, g) for g in row["genres"][:3]) if row["genres"] else "",
                "similarity": round(float(sims[i]), 4),
            })
    return result


def get_user_top_rated(user_id, n=10):
    user_ratings = ratings_df[ratings_df["user_id"] == user_id].nlargest(n, "rating")
    result = []
    for _, r in user_ratings.iterrows():
        movie = movies_df[movies_df["item_id"] == r["item_id"]]
        if not movie.empty:
            row = movie.iloc[0]
            result.append({
                "item_id": int(r["item_id"]),
                "title": row["clean_title"],
                "year": int(row["year"]) if not pd.isna(row.get("year")) else None,
                "genres": ", ".join(GENRE_CN.get(g, g) for g in row["genres"][:3]) if row["genres"] else "",
                "rating": int(r["rating"]),
            })
    return result


# ===================== 侧边栏 =====================
with st.sidebar:
    st.title("电影推荐")
    st.markdown("基于 **ALS 矩阵分解** 的协同过滤推荐引擎")
    st.markdown("---")

    st.subheader("数据概况")
    st.metric("用户数", f"{n_users:,}")
    st.metric("电影数", f"{n_items:,}")
    st.metric("评分数", f"{len(ratings_df):,}")
    sparsity = data["interactions"].nnz / (n_users * n_items) * 100
    st.metric("数据稠密度", f"{sparsity:.1f}%")

    st.markdown("---")
    st.subheader("模型配置")
    for k, v in model_config.items():
        st.caption(f"**{k}**: {v}")

    st.markdown("---")
    with st.expander("技术栈"):
        st.markdown("""
        - **数据**: MovieLens 100k
        - **算法**: ALS 交替最小二乘
        - **框架**: implicit
        - **损失**: 加权 RMSE
        - **维度**: 64 维隐向量
        - **部署**: FastAPI + Streamlit
        """)

# ===================== 主页面 =====================
st.title("电影推荐系统")
st.markdown("基于协同过滤的个性化电影推荐 — 为你发现下一部想看的电影")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "个性化推荐", "热门榜单", "搜索电影", "数据分析", "最新热门电影"
])

# ---- Tab 1: 个性化推荐 ----
with tab1:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("选择用户")
        user_id = st.number_input("用户 ID (1-943)", min_value=1, max_value=n_users, value=1)
        n_recs = st.slider("推荐数量", 5, 30, 10)
        refresh_clicked = st.button("刷新推荐", use_container_width=True)

        st.markdown("---")
        st.subheader("该用户评分最高的电影")
        top_rated = get_user_top_rated(user_id, n=5)
        if top_rated:
            for m in top_rated:
                st.markdown(f"- **{m['title']}** ({m['year']}) — {'★' * m['rating']}")
        else:
            st.caption("该用户暂无评分记录")

    with col2:
        st.subheader(f"为用户 {user_id} 推荐 {n_recs} 部电影")

        if "recs_cache" not in st.session_state:
            st.session_state.recs_cache = {}
        cache_key = (user_id, n_recs)
        if refresh_clicked or cache_key not in st.session_state.recs_cache:
            st.session_state.recs_cache[cache_key] = recommend_for_user(user_id, n=n_recs)
        recs = st.session_state.recs_cache[cache_key]

        if recs:
            cols = st.columns(2)
            for i, rec in enumerate(recs):
                with cols[i % 2]:
                    score_color = "#1f77b4" if rec["score"] > 0.5 else ("#ff7f0e" if rec["score"] > 0.3 else "#999")
                    st.markdown(f"""
                    <div style="border:1px solid #e0e0e0;border-radius:8px;padding:12px;margin-bottom:8px;
                                border-left:4px solid {score_color}" role="listitem" aria-label="{rec['title']} — 匹配度 {rec['score']:.2f}">
                        <div style="font-weight:bold;font-size:16px">{rec['title']}</div>
                        <div style="color:#666;font-size:13px">{rec.get('year', '')} · {rec['genres']}</div>
                        <div style="margin-top:6px">
                            <span style="background:{score_color};color:white;padding:2px 8px;border-radius:4px;font-size:12px">
                                匹配度 {rec['score']:.3f}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("未找到该用户的推荐数据")

# ---- Tab 2: 热门榜单 ----
with tab2:
    st.subheader("热门电影 TOP 30")
    st.markdown("按评分数量排序，反映大众认可度")

    top_n = st.slider("显示数量", 10, 50, 30, key="top_n_slider")
    popular = get_popular_movies(n=top_n)

    if popular:
        df_pop = pd.DataFrame(popular)
        df_pop["排名"] = range(1, len(df_pop) + 1)
        df_pop = df_pop[["排名", "title", "year", "genres", "avg_rating", "num_ratings"]]
        df_pop.columns = ["排名", "电影名称", "年份", "类型", "均分", "评分人数"]

        st.dataframe(
            df_pop.style.background_gradient(subset=["均分", "评分人数"], cmap="Blues")
                .format({"均分": "{:.1f}"}),
            use_container_width=True, hide_index=True, height=600,
        )

        # Chart
        fig = px.bar(
            df_pop.head(15).iloc[::-1], x="评分人数", y="电影名称", orientation="h",
            color="均分", color_continuous_scale="RdYlGn",
            title="热门电影 TOP 15",
        )
        fig.update_layout(height=450, yaxis=dict(tickfont=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: 搜索电影 ----
with tab3:
    st.subheader("搜索电影")
    query = st.text_input("输入电影名称关键词", placeholder="例如: 星球大战, 黑客帝国, 阿甘正传…")

    if query:
        results = search_movies(query)
        if results:
            st.markdown(f"找到 **{len(results)}** 部相关电影")
            cols = st.columns(3)
            for i, m in enumerate(results):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div style="border:1px solid #e0e0e0;border-radius:8px;padding:12px;margin-bottom:8px;min-height:80px" role="listitem" aria-label="{m['title']} ({m.get('year', '')})">
                        <div style="font-weight:bold">{m['title']}</div>
                        <div style="color:#666;font-size:13px">{m.get('year', '')} · {m['genres'][:60]}</div>
                        <div style="color:#999;font-size:11px">ID: {m['item_id']}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(f"未找到包含「{query}」的电影，请尝试其他关键词")

# ---- Tab 4: 数据分析 ----
with tab4:
    st.subheader("数据洞察")

    col1, col2 = st.columns(2)

    with col1:
        # 评分分布
        rating_dist = ratings_df["rating"].value_counts().sort_index()
        fig = px.bar(x=rating_dist.index, y=rating_dist.values,
                    labels={"x": "评分", "y": "数量"},
                    title="评分分布", color=rating_dist.values,
                    color_continuous_scale="Blues")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 用户评分数量分布
        user_counts = ratings_df.groupby("user_id").size()
        fig = px.histogram(x=user_counts, nbins=50,
                          labels={"x": "评分数量", "y": "用户数"},
                          title="用户评分活跃度分布",
                          color_discrete_sequence=["#636EFA"])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # 电影类型分布
    all_genres = []
    for gl in movies_df["genres"]:
        if gl: all_genres.extend(GENRE_CN.get(g, g) for g in gl)
    genre_counts = pd.Series(all_genres).value_counts()

    fig = px.bar(x=genre_counts.index, y=genre_counts.values,
                labels={"x": "类型", "y": "电影数"},
                title="电影类型分布", color=genre_counts.values,
                color_continuous_scale="Viridis")
    fig.update_layout(height=400, showlegend=False, xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 5: 最新热门电影 ----
with tab5:
    st.subheader("最新热门电影")
    st.markdown("基于 MovieLens 最新数据集，无需 API Key，直连 GroupLens")

    try:
        lm = _load_latest()
    except Exception:
        lm = None

    if lm is None:
        st.info("最新电影数据集需从国外服务器下载，当前网络不可用。其他功能（个性化推荐、热门榜单、搜索、数据分析）不受影响。")
    else:
        st.metric("数据集电影总数", f"{len(lm):,}")

        col_hot, col_recent = st.columns(2)

        with col_hot:
            st.subheader("热门综合榜")
            st.caption("评分人数多 + 均分高")
            hot = get_cached_hot_movies()
            for i, m in enumerate(hot):
                color = "#FFD700" if i < 3 else ("#BF00FF" if i < 6 else "#999")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;background:#1a1a2e;border-radius:8px;
                            padding:10px;margin-bottom:6px;border:1px solid #2a2a4e">
                    <div style="font-size:20px;font-weight:bold;color:{color};min-width:28px;text-align:center">
                        {i+1}
                    </div>
                    <div style="flex:1">
                        <div style="font-weight:bold;font-size:15px;color:#e0e0e0">{m['title']}</div>
                        <div style="color:#999;font-size:12px">{m['year'] or ''} · {m['genres']}</div>
                    </div>
                    <div style="text-align:right">
                        <div style="color:#FFD700;font-weight:bold">{m['avg_rating']:.1f}</div>
                        <div style="color:#666;font-size:11px">{m['num_ratings']:,} 评</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col_recent:
            st.subheader("近年高分榜")
            st.caption("2015年之后 + 高均分")
            recent = get_cached_recent_movies()
            for i, m in enumerate(recent):
                color = "#FFD700" if i < 3 else ("#BF00FF" if i < 6 else "#999")
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;background:#1a1a2e;border-radius:8px;
                            padding:10px;margin-bottom:6px;border:1px solid #2a2a4e">
                    <div style="font-size:20px;font-weight:bold;color:{color};min-width:28px;text-align:center">
                        {i+1}
                    </div>
                    <div style="flex:1">
                        <div style="font-weight:bold;font-size:15px;color:#e0e0e0">{m['title']}</div>
                        <div style="color:#999;font-size:12px">{m['year'] or ''} · {m['genres']}</div>
                    </div>
                    <div style="text-align:right">
                        <div style="color:#FFD700;font-weight:bold">{m['avg_rating']:.1f}</div>
                        <div style="color:#666;font-size:11px">{m['num_ratings']:,} 评</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")
st.caption("电影推荐系统 · 基于 implicit ALS · 数据科学项目集")
