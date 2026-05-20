"""
推荐系统 API 服务
基于 implicit (FastAPI) + 混合推荐（CF + 内容特征）

启动: uvicorn serve:app --host 0.0.0.0 --port 8001 --reload
"""

import os
import traceback

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_movielens_100k
from train import precision_recall_at_k


app = FastAPI(
    title="Movie Recommender API",
    description="基于 implicit 的协同过滤推荐服务 + 内容特征混合推荐",
    version="2.1.0",
)

model = None
model_config = None
data = None
item_sim_matrix = None  # 预计算的电影类型相似度矩阵

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


def _build_item_sim():
    """基于电影类型特征计算 item-item 余弦相似度矩阵。"""
    global item_sim_matrix
    if data is None or data.get("item_features") is None:
        item_sim_matrix = None
        return
    item_feat = data["item_features"].toarray()
    item_sim_matrix = cosine_similarity(item_feat)
    print(f"[混合推荐] 电影类型相似度矩阵已构建 ({item_sim_matrix.shape[0]}x{item_sim_matrix.shape[1]})")


def content_based_items(item_idx: int, n: int = 10) -> list[int]:
    """为冷门电影找到类型最相似的 N 部电影。"""
    if item_sim_matrix is None or item_idx >= item_sim_matrix.shape[0]:
        return []
    sims = item_sim_matrix[item_idx]
    top = np.argsort(-sims)[1:n+1]  # 跳过自己
    return [(int(i), float(sims[i])) for i in top if sims[i] > 0]


@app.on_event("startup")
def startup():
    global model, model_config, data
    print("[Startup] Loading data...")
    data = load_movielens_100k()
    _build_item_sim()

    candidates = sorted(
        [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")],
        reverse=True,
    )
    if not candidates:
        print("[Startup] No model found, quick-training...")
        _quick_train()
        candidates = sorted(
            [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")],
            reverse=True,
        )

    model_path = os.path.join(MODEL_DIR, candidates[0])
    print(f"[Startup] Loading model: {model_path}")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    model_config = bundle["config"]
    print(f"[Startup] Model ready | config: {model_config}")


def _quick_train():
    """Quick-train a default model"""
    from data_loader import build_implicit_feedback
    from implicit.als import AlternatingLeastSquares

    interactions = build_implicit_feedback(data, threshold=4.0)
    model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=20, random_state=42, num_threads=4)
    model.fit(interactions, show_progress=False)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "implicit_als_64d.pkl")
    joblib.dump({"model": model, "config": {
        "algorithm": "als", "factors": 64, "iterations": 20, "threshold": 4.0,
    }}, model_path)


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[dict] = Field(..., description="Recommended movies")


class SearchResponse(BaseModel):
    query: str
    results: list[dict] = Field(..., description="Search results")


class PopularResponse(BaseModel):
    movies: list[dict] = Field(..., description="Popular movies")


def recommend_for_user(user_id: int, n: int = 10) -> list[dict]:
    """Generate Top-N recommendations for a user"""
    if model is None or data is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    n_users = data["n_users"]
    user_idx = user_id - 1

    if user_idx < 0 or user_idx >= n_users:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found (range: 1-{n_users})")

    interactions = data["interactions"]
    user_items = interactions[user_idx]

    recs = model.recommend(user_idx, user_items, N=n, filter_already_liked_items=True)
    item_indices = recs[0]
    scores = recs[1]

    recommendations = []
    movies_df = data["movies"]
    for item_idx, score in zip(item_indices, scores):
        item_id = int(item_idx) + 1
        movie = movies_df[movies_df["item_id"] == item_id]
        if not movie.empty:
            row = movie.iloc[0]
            recommendations.append({
                "item_id": item_id,
                "title": row["clean_title"],
                "year": int(row["year"]) if not pd.isna(row.get("year")) else None,
                "genres": row["genres"],
                "score": round(float(score), 4),
            })

    return recommendations


def get_popular_movies(n: int = 20) -> list[dict]:
    """Get popular movies by rating count"""
    ratings = data["ratings"]
    popular = (ratings.groupby("item_id")
               .agg(count=("rating", "count"), avg_rating=("rating", "mean"))
               .sort_values("count", ascending=False)
               .head(n))

    movies_df = data["movies"]
    result = []
    for item_id, row in popular.iterrows():
        movie = movies_df[movies_df["item_id"] == item_id]
        if not movie.empty:
            result.append({
                "item_id": int(item_id),
                "title": movie.iloc[0]["clean_title"],
                "year": int(movie.iloc[0]["year"]) if not pd.isna(movie.iloc[0].get("year")) else None,
                "genres": movie.iloc[0]["genres"],
                "avg_rating": round(float(row["avg_rating"]), 2),
                "num_ratings": int(row["count"]),
            })
    return result


def search_movies(query: str, limit: int = 10) -> list[dict]:
    """Search movies by title"""
    movies_df = data["movies"]
    mask = movies_df["clean_title"].str.contains(query, case=False, na=False)
    results = movies_df[mask].head(limit)
    return [
        {
            "item_id": int(row["item_id"]),
            "title": row["clean_title"],
            "year": int(row["year"]) if not pd.isna(row["year"]) else None,
            "genres": row["genres"],
        }
        for _, row in results.iterrows()
    ]


# ---------- API Routes ----------
@app.get("/", response_model=dict)
def root():
    return {
        "service": "Movie Recommender API",
        "status": "ok" if model else "degraded",
        "model_config": model_config,
        "endpoints": {
            "GET /recommend/{user_id}": "CF 推荐电影",
            "GET /recommend/{user_id}/hybrid": "混合推荐 (CF + 内容特征)",
            "GET /popular": "热门电影",
            "GET /search?q=title": "搜索电影",
            "GET /movie/{item_id}": "电影详情",
            "GET /similar/{item_id}": "相似电影 (基于类型)",
            "GET /docs": "Swagger UI",
        },
    }


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, n: int = Query(default=10, le=50)):
    """Generate Top-N movie recommendations for a user"""
    recs = recommend_for_user(user_id, n=n)
    return RecommendResponse(user_id=user_id, recommendations=recs)


@app.get("/popular", response_model=PopularResponse)
def popular(n: int = Query(default=20, le=100)):
    """Get popular movies leaderboard"""
    movies = get_popular_movies(n)
    return PopularResponse(movies=movies)


@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1), limit: int = Query(default=10, le=50)):
    """Search movies"""
    results = search_movies(q, limit=limit)
    return SearchResponse(query=q, results=results)


@app.get("/movie/{item_id}")
def movie_detail(item_id: int):
    """Get movie details"""
    movies_df = data["movies"]
    movie = movies_df[movies_df["item_id"] == item_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail=f"Movie {item_id} not found")

    row = movie.iloc[0]
    return {
        "item_id": int(row["item_id"]),
        "title": row["clean_title"],
        "year": int(row["year"]) if not pd.isna(row["year"]) else None,
        "genres": row["genres"],
    }


@app.get("/recommend/{user_id}/hybrid")
def recommend_hybrid(user_id: int, n: int = Query(default=10, le=50), cb_weight: float = Query(default=0.2, ge=0, le=1)):
    """混合推荐：CF 为主，内容相似度为冷门物品兜底"""
    cf_recs = recommend_for_user(user_id, n=n)
    if item_sim_matrix is None or not cf_recs:
        return RecommendResponse(user_id=user_id, recommendations=cf_recs)

    # 找到用户喜欢的物品（训练集中评分>=4的物品）
    user_idx = user_id - 1
    user_items = data["interactions"][user_idx]
    liked = set(int(i) for i in user_items.indices if user_items[0, i] >= 4)

    boosted = []
    for rec in cf_recs:
        item_idx = rec["item_id"] - 1
        cb_score = 0.0
        if liked:
            sims = item_sim_matrix[item_idx]
            cb_score = float(np.mean([sims[li] for li in liked if li < len(sims)] or [0]))
        hybrid_score = (1 - cb_weight) * rec["score"] + cb_weight * cb_score
        boosted.append({**rec, "score": round(hybrid_score, 4), "cb_score": round(cb_score, 4)})

    boosted.sort(key=lambda x: -x["score"])
    return RecommendResponse(user_id=user_id, recommendations=boosted[:n])


@app.get("/similar/{item_id}")
def similar_movies(item_id: int, n: int = Query(default=10, le=20)):
    """基于电影类型找到相似的电影（内容推荐）"""
    if item_sim_matrix is None:
        raise HTTPException(status_code=503, detail="内容特征不可用")
    item_idx = item_id - 1
    if item_idx < 0 or item_idx >= item_sim_matrix.shape[0]:
        raise HTTPException(status_code=404, detail=f"Movie {item_id} not found")

    sims = item_sim_matrix[item_idx]
    top = np.argsort(-sims)[1:n+1]
    movies_df = data["movies"]
    results = []
    for i in top:
        movie = movies_df[movies_df["item_id"] == i + 1]
        if not movie.empty:
            row = movie.iloc[0]
            results.append({
                "item_id": int(i + 1),
                "title": row["clean_title"],
                "year": int(row["year"]) if not pd.isna(row.get("year")) else None,
                "genres": row["genres"],
                "similarity": round(float(sims[i]), 4),
            })
    return {"item_id": item_id, "similar": results}


@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": traceback.format_exc()},
    )
