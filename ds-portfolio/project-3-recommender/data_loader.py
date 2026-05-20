"""
数据加载与预处理模块
支持 MovieLens 100k / 1M 数据集
基于 implicit 库
"""

import os
import zipfile
import urllib.request

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


# ---------- MovieLens 数据集下载 ----------

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_LATEST_URL = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def download_movielens(size: str = "100k") -> str:
    """下载 MovieLens 数据集并解压"""
    urls = {"100k": MOVIELENS_100K_URL, "1m": MOVIELENS_1M_URL}
    url = urls.get(size, MOVIELENS_100K_URL)
    os.makedirs(DATA_DIR, exist_ok=True)

    zip_name = f"ml-{size}.zip"
    zip_path = os.path.join(DATA_DIR, zip_name)
    extract_path = os.path.join(DATA_DIR, f"ml-{size}")

    if os.path.exists(extract_path):
        print(f"[OK] 数据集已存在: {extract_path}")
        return extract_path

    print(f"[下载] 正在下载 MovieLens {size}...")
    urllib.request.urlretrieve(url, zip_path)
    print(f"[OK] 下载完成")

    print("[解压] 正在解压...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"[OK] 解压完成: {extract_path}")
    return extract_path


def _generate_demo_data(n_users=100, n_items=200, n_ratings=5000, seed=42) -> dict:
    """生成演示用模拟电影评分数据。下载失败时兜底。"""
    rng = np.random.default_rng(seed)
    movie_titles = [
        "The Matrix", "Inception", "Interstellar", "The Dark Knight", "Pulp Fiction",
        "Fight Club", "Forrest Gump", "The Shawshank Redemption", "The Godfather",
        "Schindler's List", "Goodfellas", "The Silence of the Lambs", "Star Wars",
        "Jurassic Park", "Toy Story", "Finding Nemo", "The Lion King", "Frozen",
        "The Avengers", "Iron Man", "Spider-Man", "Black Panther", "Avatar",
        "Titanic", "Gladiator", "Braveheart", "The Lord of the Rings", "Harry Potter",
        "The Social Network", "Whiplash", "La La Land", "Parasite", "Joker",
        "The Grand Budapest Hotel", "Mad Max: Fury Road", "Blade Runner 2049",
        "Dune", "Everything Everywhere All at Once", "Oppenheimer", "Barbie",
    ] * 5  # Repeat to reach 200
    genre_names = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
                   "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller", "War"]
    movies_data = []
    for i in range(n_items):
        year = rng.integers(1970, 2025)
        n_genres = rng.integers(1, 4)
        genres = list(rng.choice(genre_names, n_genres, replace=False))
        movies_data.append({
            "item_id": i + 1,
            "title": movie_titles[i],
            "year": year,
            "clean_title": movie_titles[i],
            "genres": genres,
            **{g: 1 if g in genres else 0 for g in genre_names},
        })

    movies = pd.DataFrame(movies_data)
    missing_cols = [c for c in ["release_date", "video_release_date", "IMDb_URL"] if c not in movies.columns]
    for c in missing_cols:
        movies[c] = ""

    ratings_data = []
    for _ in range(n_ratings):
        ratings_data.append({
            "user_id": int(rng.integers(1, n_users + 1)),
            "item_id": int(rng.integers(1, n_items + 1)),
            "rating": int(rng.integers(1, 6)),
            "timestamp": int(rng.integers(800_000_000, 900_000_000)),
        })
    ratings = pd.DataFrame(ratings_data).drop_duplicates(subset=["user_id", "item_id"])
    ratings["user_idx"] = ratings["user_id"] - 1
    ratings["item_idx"] = ratings["item_id"] - 1

    users = pd.DataFrame({
        "user_id": range(1, n_users + 1),
        "age": rng.integers(18, 65, n_users),
        "gender": rng.choice(["M", "F"], n_users),
        "occupation": "other",
        "zip_code": "00000",
    })

    interactions = coo_matrix(
        (ratings["rating"].values, (ratings["user_idx"].values, ratings["item_idx"].values)),
        shape=(n_users, n_items),
    ).tocsr()

    genre_cols = genre_names
    genre_matrix = movies[genre_cols].values.astype(np.float32)
    item_features = csr_matrix(genre_matrix)

    print(f"\n[数据] 演示数据已生成 (网络下载失败，使用模拟数据)")
    print(f"  - 用户数: {n_users} | 电影数: {n_items} | 评分数: {len(ratings)}")
    return {
        "ratings": ratings, "movies": movies, "users": users,
        "interactions": interactions, "n_users": n_users, "n_items": n_items,
        "item_features": item_features,
    }


def load_movielens_100k(data_dir: str = None) -> dict:
    """加载 MovieLens 100k 数据集，下载失败自动回退演示数据。"""
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, "ml-100k")

    ratings_path = os.path.join(data_dir, "u.data")
    if not os.path.exists(ratings_path):
        try:
            data_dir = download_movielens("100k")
            ratings_path = os.path.join(data_dir, "u.data")
        except Exception as e:
            print(f"[提示] MovieLens 下载失败 ({e})，使用演示数据")
            return _generate_demo_data()

    if not os.path.exists(ratings_path):
        return _generate_demo_data()

    ratings_cols = ["user_id", "item_id", "rating", "timestamp"]
    ratings = pd.read_csv(ratings_path, sep="\t", names=ratings_cols)

    movies_path = os.path.join(data_dir, "u.item")
    movies_cols = [
        "item_id", "title", "release_date", "video_release_date",
        "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama",
        "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western",
    ]
    movies = pd.read_csv(movies_path, sep="|", encoding="latin-1", names=movies_cols)

    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)

    genre_cols = [
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western",
    ]
    movies["genres"] = movies[genre_cols].apply(
        lambda row: [col for col in genre_cols if row[col] == 1], axis=1
    )

    users_path = os.path.join(data_dir, "u.user")
    users_cols = ["user_id", "age", "gender", "occupation", "zip_code"]
    users = pd.read_csv(users_path, sep="|", names=users_cols)

    ratings["user_idx"] = ratings["user_id"] - 1
    ratings["item_idx"] = ratings["item_id"] - 1

    n_users = ratings["user_id"].max()
    n_items = ratings["item_id"].max()

    interactions = coo_matrix(
        (ratings["rating"].values,
         (ratings["user_idx"].values, ratings["item_idx"].values)),
        shape=(n_users, n_items),
    ).tocsr()

    item_features = None
    if all(c in movies.columns for c in genre_cols):
        genre_matrix = movies[genre_cols].values.astype(np.float32)
        item_features = csr_matrix(genre_matrix)

    result = {
        "ratings": ratings,
        "movies": movies,
        "users": users,
        "interactions": interactions,
        "n_users": n_users,
        "n_items": n_items,
        "item_features": item_features,
    }

    print(f"\n[数据] MovieLens 100k 数据集加载完成")
    print(f"  - 用户数: {n_users}")
    print(f"  - 电影数: {n_items}")
    print(f"  - 评分数: {len(ratings)}")
    print(f"  - 稀疏度: {interactions.nnz / (n_users * n_items):.4%}")
    if item_features is not None:
        print(f"  - 电影特征维度: {item_features.shape[1]} (类型)")

    return result


def build_implicit_feedback(data: dict, threshold: float = 4.0) -> csr_matrix:
    """将显式评分转为隐式反馈 (评分 >= threshold 视为正反馈)"""
    interactions = data["interactions"].copy()
    interactions.data = (interactions.data >= threshold).astype(np.float32)
    interactions.eliminate_zeros()
    return interactions


def load_latest_movies() -> pd.DataFrame:
    """
    下载并加载 MovieLens 最新版数据集 (ml-latest, ~250MB)，
    返回包含最新热门电影信息的 DataFrame。
    优先 ml-latest（含 2023 年电影），下载失败回退到 ml-latest-small。
    无需 API Key，直连 GroupLens 下载。
    """
    import zipfile, io

    latest_dir = None
    for version in [("ml-latest", MOVIELENS_LATEST_URL, "2023"),
                    ("ml-latest-small", MOVIELENS_LATEST_SMALL_URL, "2018")]:
        candidate_dir = os.path.join(DATA_DIR, version[0])
        if os.path.exists(candidate_dir):
            latest_dir = candidate_dir
            print(f"[OK] 数据集已存在: {latest_dir}（年份到 {version[2]}）")
            break
        try:
            print(f"[下载] 正在下载 MovieLens {version[0]}...")
            os.makedirs(DATA_DIR, exist_ok=True)
            zip_path = os.path.join(DATA_DIR, f"{version[0]}.zip")
            urllib.request.urlretrieve(version[1], zip_path)
            print("[解压] 正在解压...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DATA_DIR)
            latest_dir = candidate_dir
            print(f"[OK] 解压完成: {latest_dir}（年份到 {version[2]}）")
            break
        except Exception as e:
            print(f"[提示] {version[0]} 下载失败 ({e})，尝试下一个...")
            continue

    if latest_dir is None:
        raise RuntimeError("所有 MovieLens 下载源均不可用（网络受限），跳过最新电影数据")

    movies_path = os.path.join(latest_dir, "movies.csv")
    ratings_path = os.path.join(latest_dir, "ratings.csv")

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    # 提取年份
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)

    # 评分统计
    stats = (ratings.groupby("movieId")
             .agg(rating_count=("rating", "count"), rating_mean=("rating", "mean"))
             .reset_index())

    movies = movies.merge(stats, on="movieId", how="left")
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)
    movies["rating_mean"] = movies["rating_mean"].fillna(0).round(1)

    # 转换类型名
    genre_cn = {
        "Action": "动作", "Adventure": "冒险", "Animation": "动画", "Children": "儿童",
        "Comedy": "喜剧", "Crime": "犯罪", "Documentary": "纪录", "Drama": "剧情",
        "Fantasy": "奇幻", "Film-Noir": "黑色电影", "Horror": "恐怖", "Musical": "音乐",
        "Mystery": "悬疑", "Romance": "爱情", "Sci-Fi": "科幻", "Thriller": "惊悚",
        "War": "战争", "Western": "西部", "IMAX": "巨幕",
    }
    movies["genres_cn"] = movies["genres"].apply(
        lambda s: [genre_cn.get(g, g) for g in str(s).split("|") if g != "(no genres listed)"]
    )

    print(f"[数据] ml-latest-small 加载完成")
    print(f"  - 电影数: {len(movies)}")
    print(f"  - 评分记录: {len(ratings):,}")
    print(f"  - 年份范围: {movies['year'].min():.0f} ~ {movies['year'].max():.0f}")

    return movies


def get_hot_movies(latest_movies: pd.DataFrame, min_ratings: int = 50, top_n: int = 20) -> list:
    """从最新数据集中提取热门电影（高评分数量 + 高均分）"""
    df = latest_movies[latest_movies["rating_count"] >= min_ratings].copy()
    df["hot_score"] = df["rating_count"] * df["rating_mean"]
    df = df.sort_values("hot_score", ascending=False).head(top_n)
    return [{
        "title": row["clean_title"],
        "year": int(row["year"]) if not pd.isna(row["year"]) else None,
        "genres": ", ".join(row["genres_cn"][:3]) if row["genres_cn"] else "",
        "avg_rating": float(f"{row['rating_mean']:.1f}"),
        "num_ratings": int(row["rating_count"]),
    } for _, row in df.iterrows()]


def get_recent_movies(latest_movies: pd.DataFrame, min_ratings: int = 30, top_n: int = 20) -> list:
    """从最新数据集中提取近期高分电影（2015+，高均分）"""
    df = latest_movies[
        (latest_movies["rating_count"] >= min_ratings)
        & (latest_movies["year"] >= 2015)
    ].copy()
    df = df.sort_values("rating_mean", ascending=False).head(top_n)
    return [{
        "title": row["clean_title"],
        "year": int(row["year"]) if not pd.isna(row["year"]) else None,
        "genres": ", ".join(row["genres_cn"][:3]) if row["genres_cn"] else "",
        "avg_rating": float(f"{row['rating_mean']:.1f}"),
        "num_ratings": int(row["rating_count"]),
    } for _, row in df.iterrows()]


def train_test_split_interactions(interactions, test_percentage=0.2, random_state=42):
    """将交互矩阵拆分为训练/测试集 (随机屏蔽 test_percentage 的交互)"""
    rng = np.random.RandomState(random_state)
    train = interactions.copy().tocoo()
    test = coo_matrix(interactions.shape, dtype=np.float32)

    nnz = train.nnz
    mask = rng.rand(nnz) < test_percentage

    test_row = train.row[mask]
    test_col = train.col[mask]
    test_data = train.data[mask]

    train.data[mask] = 0
    train.eliminate_zeros()

    test = coo_matrix(
        (test_data, (test_row, test_col)),
        shape=interactions.shape,
    ).tocsr()

    return train.tocsr(), test
