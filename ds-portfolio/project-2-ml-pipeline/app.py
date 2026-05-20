"""
ML 文本分类系统 — Streamlit 交互界面
运行: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import json
import os
import requests
from data import clean_text

st.set_page_config(
    page_title="新闻文本智能分类系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- 加载模型 ----
@st.cache_resource
def load_model():
    import config
    from predict import load_artifacts
    model, extractor = load_artifacts()
    return model, extractor, list(model.classes_)

with st.spinner("加载模型中…"):
    model, extractor, labels = load_model()

# @st.cache_data is replaced by the predict function below
def classify(text):
    cleaned = clean_text(text)
    X = extractor.transform(np.array([cleaned]))
    pred = model.predict(X)[0]
    probs = None
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(X)[0]
        probs = {cls: float(p) for cls, p in zip(model.classes_, proba_arr)}
    return pred, probs


def fetch_live_news(api_key, country="us", limit=15):
    """从 NewsAPI 获取实时新闻头条"""
    url = "https://newsapi.org/v2/top-headlines"
    params = {"country": country, "apiKey": api_key, "pageSize": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return [{"title": a["title"], "description": a.get("description", "")} for a in articles if a.get("title")]

# ===================== 侧边栏 =====================
with st.sidebar:
    st.title("新闻文本分类")
    st.markdown("基于 **TF-IDF + SVD + 逻辑回归** 的新闻文本多分类系统")
    st.markdown("---")

    st.subheader("模型信息")
    st.metric("类别数", len(labels))
    st.metric("特征维度", "5,000 (TF-IDF) → 100 (SVD)")

    st.markdown("---")
    st.subheader("分类类别")
    for label in labels:
        st.markdown(f"- **{label}**")

    st.markdown("---")
    # NewsAPI 配置
    with st.expander("实时新闻配置"):
        news_api_key = st.text_input(
            "NewsAPI Key",
            value=os.environ.get("NEWSAPI_KEY", ""),
            type="password",
            placeholder="在此粘贴 NewsAPI Key（免费注册: newsapi.org）",
            help="留空则使用示例新闻数据"
        )
        news_country = st.selectbox("新闻来源", ["us", "cn", "jp", "de", "fr"], index=0)

    st.markdown("---")
    with st.expander("技术栈"):
        st.markdown("""
        - **数据**: 20 Newsgroups (17,886 篇)
        - **清洗**: 正则 + 停用词
        - **特征**: TF-IDF (ngram 1-2) → TruncatedSVD
        - **模型**: 逻辑回归 (F1=0.81)
        - **追踪**: MLflow 实验管理
        - **部署**: FastAPI + Docker
        """)

# ===================== 主页面 =====================
st.title("新闻文本智能分类系统")
st.markdown("输入任意英文新闻文本，智能识别所属类别：计算机、科学、哲学、社科、休闲")
st.markdown("---")

# ---- 输入区域 ----
col_input, col_result = st.columns([3, 2])

with col_input:
    st.subheader("输入文本")

    # 快捷示例
    st.caption("或选择一个预设示例：")
    examples = {
        "自定义输入…": "",
        "计算机硬件": "The new graphics card from NVIDIA offers incredible performance for gaming and artificial intelligence workloads with its advanced CUDA cores.",
        "太空探索": "NASA scientists have discovered a new exoplanet in the habitable zone of a nearby star system using the James Webb Space Telescope.",
        "体育赛事": "The baseball team won their final game of the season with a spectacular home run in the bottom of the ninth inning.",
        "政治讨论": "The president addressed Congress today to discuss foreign policy and immigration reform while facing criticism from opposition leaders.",
        "宗教哲学": "The existence of God and the nature of faith have been debated by philosophers and theologians for centuries.",
    }

    selected = st.selectbox("选择预设示例", list(examples.keys()))

    user_text = st.text_area(
        "输入英文新闻文本",
        value=examples[selected],
        height=180,
        placeholder="在此输入或粘贴英文新闻文本…",
        label_visibility="collapsed" if selected != "自定义输入…" else "visible",
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        classify_btn = st.button("开始分类", type="primary", use_container_width=True)
    with col_btn2:
        fetch_news_btn = st.button("获取实时新闻", use_container_width=True)

    if fetch_news_btn:
        if news_api_key:
            with st.spinner("获取实时新闻中…"):
                try:
                    live_articles = fetch_live_news(news_api_key, news_country, limit=10)
                    st.session_state.live_news = live_articles
                    st.success(f"已获取 {len(live_articles)} 条新闻头条")
                except Exception as e:
                    err_msg = str(e)
                    if "401" in err_msg or "403" in err_msg:
                        st.warning("NewsAPI Key 无效或免费额度已用完。NewsAPI 自 2024 年起限制免费访问。以下使用预设示例演示。")
                        demo_articles = [
                            {"title": "NASA's James Webb Telescope discovers Earth-like exoplanet in habitable zone", "description": "Scientists announced the groundbreaking discovery of a new planet that could potentially support life."},
                            {"title": "Stock market hits all-time high as tech earnings exceed expectations", "description": "Major indices surged today following strong quarterly reports from leading technology companies."},
                            {"title": "World Cup final ends in dramatic penalty shootout victory", "description": "The championship match went into extra time before being decided by penalties."},
                            {"title": "Supreme Court to hear landmark case on digital privacy rights", "description": "The case could set significant legal precedent for how technology companies handle user data."},
                            {"title": "Breakthrough in quantum computing could revolutionize drug discovery", "description": "Researchers demonstrated a new quantum processor capable of solving complex molecular simulations."},
                            {"title": "Major League Baseball season preview: Top teams and players to watch", "description": "Opening day is just around the corner as teams finalize their rosters."},
                            {"title": "Philosophers debate ethics of artificial intelligence at global summit", "description": "Leading thinkers gathered to discuss the moral implications of advanced AI systems."},
                            {"title": "New GPU architecture promises 2x performance for deep learning workloads", "description": "The latest hardware generation targets AI training and inference acceleration."},
                        ]
                        st.session_state.live_news = demo_articles
                    else:
                        st.error(f"获取失败: {e}")
        else:
            # 无 API Key 时用预设示例模拟实时新闻体验
            demo_articles = [
                {"title": "NASA's James Webb Telescope discovers Earth-like exoplanet in habitable zone", "description": "Scientists announced the groundbreaking discovery of a new planet that could potentially support life."},
                {"title": "Stock market hits all-time high as tech earnings exceed expectations", "description": "Major indices surged today following strong quarterly reports from leading technology companies."},
                {"title": "World Cup final ends in dramatic penalty shootout victory", "description": "The championship match went into extra time before being decided by penalties."},
                {"title": "Supreme Court to hear landmark case on digital privacy rights", "description": "The case could set significant legal precedent for how technology companies handle user data."},
                {"title": "Breakthrough in quantum computing could revolutionize drug discovery", "description": "Researchers demonstrated a new quantum processor capable of solving complex molecular simulations."},
                {"title": "Major League Baseball season preview: Top teams and players to watch", "description": "Opening day is just around the corner as teams finalize their rosters."},
                {"title": "Philosophers debate ethics of artificial intelligence at global summit", "description": "Leading thinkers gathered to discuss the moral implications of advanced AI systems."},
                {"title": "New GPU architecture promises 2x performance for deep learning workloads", "description": "The latest hardware generation targets AI training and inference acceleration."},
            ]
            st.session_state.live_news = demo_articles
            st.success(f"已加载 {len(demo_articles)} 条示例新闻（配置 NewsAPI Key 可获取真实实时新闻）")

    # 展示实时新闻分类结果
    if "live_news" in st.session_state and st.session_state.live_news:
        st.markdown("---")
        st.subheader("实时新闻分类结果")
        news_results = []
        for article in st.session_state.live_news[:15]:
            text = f"{article['title']}. {article.get('description', '')}"[:500]
            pred, probs = classify(text)
            news_results.append({
                "标题": article["title"][:80],
                "预测类别": pred,
                "置信度": f"{max(probs.values()):.1%}" if probs else "N/A",
            })
        st.dataframe(pd.DataFrame(news_results), use_container_width=True, hide_index=True)

# ---- 结果展示 ----
with col_result:
    st.subheader("分类结果")

    if classify_btn and user_text.strip():
        with st.spinner("分析中…"):
            pred, probs = classify(user_text)

        # 大号结果
        color_map = {"计算机": "#1f77b4", "科学": "#2ca02c", "哲学": "#9467bd",
                     "社科": "#d62728", "休闲": "#ff7f0e", "其他": "#7f7f7f"}

        st.markdown(f"""
        <div style="background:{color_map.get(pred, '#666')};border-radius:12px;padding:20px;text-align:center;color:white" role="status" aria-live="polite">
            <div style="font-size:32px;font-weight:bold">{pred}</div>
            <div style="font-size:14px;opacity:0.85;margin-top:8px">置信度 {max(probs.values()) if probs else 0:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        if probs:
            # 概率条形图
            df_probs = pd.DataFrame({
                "类别": list(probs.keys()),
                "概率": list(probs.values()),
            }).sort_values("概率", ascending=True)

            fig = px.bar(
                df_probs, x="概率", y="类别", orientation="h",
                color="类别",
                color_discrete_map={k: color_map.get(k, "#999") for k in probs},
                text=df_probs["概率"].apply(lambda x: f"{x:.1%}"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                showlegend=False, height=250,
                margin=dict(l=20, r=50, t=10, b=10),
                xaxis=dict(range=[0, 1], tickformat=".0%"),
            )
            st.plotly_chart(fig, use_container_width=True)
    elif not classify_btn:
        st.info("输入文本后点击「开始分类」查看结果")
    elif not user_text.strip():
        st.warning("请输入文本内容后点击「开始分类」")

st.markdown("---")

# ---- 下方：模型统计 + 批量测试 ----
tab1, tab2 = st.tabs(["模型性能", "批量测试"])

with tab1:
    col1, col2, col3 = st.columns(3)

    # Try loading results
    results_path = "results/evaluation_results.json"
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)
        overall = results.get("overall", {})
        col1.metric("准确率", f"{overall.get('accuracy', 0):.1%}")
        col2.metric("F1 分数", f"{overall.get('f1_weighted', 0):.1%}")
        col3.metric("样本数", overall.get("num_samples", 0))

        st.subheader("各类别 F1 分数")
        per_class = results.get("per_class", {})
        if per_class:
            df_f1 = pd.DataFrame([
                {"类别": k, "F1分数": v["f1"], "精确率": v["precision"], "召回率": v["recall"]}
                for k, v in per_class.items()
            ])
            fig = px.bar(df_f1, x="类别", y=["F1分数", "精确率", "召回率"],
                        barmode="group", height=350,
                        color_discrete_sequence=["#636EFA", "#00CC96", "#EF553B"])
            st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    cm_path = "results/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.subheader("混淆矩阵")
        st.image(cm_path)

with tab2:
    st.subheader("批量测试")
    st.markdown("一次性测试多条文本，快速验证模型效果")

    batch_text = st.text_area(
        "输入多条文本",
        height=150,
        placeholder="The computer hardware is amazing.\nScientists found a new planet.\nThe baseball game was exciting.",
    )

    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None

    if st.button("批量分类", type="primary") and batch_text.strip():
        texts = [t.strip() for t in batch_text.split("\n") if t.strip()]
        results_list = []
        progress = st.progress(0, "批量分类中…")
        for i, t in enumerate(texts):
            pred, probs = classify(t)
            results_list.append({
                "文本": t[:100] + ("…" if len(t) > 100 else ""),
                "预测类别": pred,
                "置信度": f"{max(probs.values()):.1%}" if probs else "N/A",
            })
            progress.progress((i + 1) / len(texts), f"已处理 {i + 1}/{len(texts)}")
        progress.empty()
        st.session_state.batch_results = pd.DataFrame(results_list)

    if st.session_state.batch_results is not None:
        st.markdown(f"共 **{len(st.session_state.batch_results)}** 条结果")
        st.dataframe(st.session_state.batch_results, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("新闻文本智能分类系统 · 基于 scikit-learn + TF-IDF · 数据科学项目集")
