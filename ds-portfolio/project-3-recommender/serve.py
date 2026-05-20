"""
推荐系统 API 服务
基于 implicit (FastAPI)

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

from data_loader import load_movielens_100k
from train import precision_recall_at_k


app = FastAPI(
    title="Movie Recommender API",
    description="基于 implicit 的协同过滤推荐服务",
    version="2.0.0",
)

model = None
model_config = None
data = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


@app.on_event("startup")
def startup():
    global model, model_config, data
    print("[Startup] Loading data...")
    data = load_movielens_100k()

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
            "GET /recommend/{user_id}": "Recommend movies for user",
            "GET /popular": "Popular movies",
            "GET /search?q=title": "Search movies",
            "GET /movie/{item_id}": "Movie details",
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


@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": traceback.format_exc()},
    )
