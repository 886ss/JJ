"""
推荐模型训练模块
基于 implicit 实现协同过滤推荐
支持: als (等价 WARP), bpr (等价 BPR), lmfa (等价 Logistic)
"""

import argparse
import json
import os

import numpy as np
from scipy.sparse import csr_matrix

from data_loader import (
    load_movielens_100k,
    build_implicit_feedback,
    train_test_split_interactions,
)

# 尝试导入 implicit 的各种算法
try:
    from implicit.als import AlternatingLeastSquares
    HAS_IMPLICIT = True
except ImportError:
    HAS_IMPLICIT = False

try:
    from implicit.bpr import BayesianPersonalizedRanking
    HAS_BPR = True
except ImportError:
    HAS_BPR = False

try:
    from implicit.lmf import LogisticMatrixFactorization
    HAS_LMF = True
except ImportError:
    HAS_LMF = False


MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def precision_recall_at_k(model, train_mat, test_mat, k=10):
    """计算 Precision@K 和 Recall@K"""
    n_users = test_mat.shape[0]
    precisions = []
    recalls = []

    for user_idx in range(n_users):
        test_items = set(test_mat[user_idx].indices)
        if len(test_items) == 0:
            continue

        # 获取推荐
        recs = model.recommend(
            user_idx, train_mat[user_idx], N=k, filter_already_liked_items=True
        )
        recommended = set(recs[0])

        hits = len(recommended & test_items)
        precisions.append(hits / k)
        recalls.append(hits / len(test_items))

    return np.mean(precisions), np.mean(recalls)


def auc_score_manual(model, train_mat, test_mat):
    """简化的 AUC 计算: 对每个用户，比较测试物品 vs 随机未交互物品的得分"""
    user_factors = model.user_factors
    item_factors = model.item_factors
    n_users = test_mat.shape[0]
    n_items = test_mat.shape[1]

    aucs = []
    rng = np.random.RandomState(42)

    for user_idx in range(n_users):
        test_items = test_mat[user_idx].indices
        if len(test_items) == 0:
            continue

        train_items = set(train_mat[user_idx].indices)
        all_unseen = [i for i in range(n_items) if i not in train_items and i not in test_items]

        for test_item in test_items:
            if len(all_unseen) == 0:
                continue
            neg_item = rng.choice(all_unseen)
            score_pos = np.dot(user_factors[user_idx], item_factors[test_item])
            score_neg = np.dot(user_factors[user_idx], item_factors[neg_item])
            if score_pos > score_neg:
                aucs.append(1.0)
            elif score_pos == score_neg:
                aucs.append(0.5)
            else:
                aucs.append(0.0)

    return np.mean(aucs) if aucs else 0.0


def train(
    algorithm: str = "als",
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 30,
    threshold: float = 4.0,
    seed: int = 42,
):
    """训练推荐模型"""
    print("\n" + "=" * 60)
    print(f"Training Recommender: algo={algorithm}, factors={factors}, iters={iterations}")
    print("=" * 60)

    data = load_movielens_100k()
    interactions_implicit = build_implicit_feedback(data, threshold)
    train_mat, test_mat = train_test_split_interactions(
        interactions_implicit, test_percentage=0.2, random_state=seed
    )

    print(f"\n[训练] 正反馈样本数: {train_mat.nnz}")
    print(f"[训练] 测试样本数: {test_mat.nnz}")

    # ---------- 选择算法 ----------
    if algorithm == "als":
        if not HAS_IMPLICIT:
            raise ImportError("implicit not installed")
        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=seed,
            num_threads=4,
        )
    elif algorithm == "bpr":
        if not HAS_BPR:
            raise ImportError("implicit.bpr not available")
        model = BayesianPersonalizedRanking(
            factors=factors,
            learning_rate=0.05,
            iterations=iterations,
            random_state=seed,
            num_threads=4,
        )
    elif algorithm == "lmf":
        if not HAS_LMF:
            raise ImportError("implicit.lmf not available")
        model = LogisticMatrixFactorization(
            factors=factors,
            learning_rate=0.05,
            iterations=iterations,
            random_state=seed,
            num_threads=4,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"\n[训练] Starting ({iterations} iterations)...")
    model.fit(train_mat, show_progress=True)

    # ---------- 评估 ----------
    print("\n[评估] Computing metrics...")
    prec5, rec5 = precision_recall_at_k(model, train_mat, test_mat, k=5)
    prec10, rec10 = precision_recall_at_k(model, train_mat, test_mat, k=10)
    prec20, rec20 = precision_recall_at_k(model, train_mat, test_mat, k=20)
    auc = auc_score_manual(model, train_mat, test_mat)

    print(f"  Precision@5:  {prec5:.4f}   Recall@5:  {rec5:.4f}")
    print(f"  Precision@10: {prec10:.4f}   Recall@10: {rec10:.4f}")
    print(f"  Precision@20: {prec20:.4f}   Recall@20: {rec20:.4f}")
    print(f"  AUC:          {auc:.4f}")

    # ---------- 保存模型 ----------
    import joblib
    model_path = os.path.join(MODEL_DIR, f"implicit_{algorithm}_{factors}d.pkl")
    joblib.dump({
        "model": model,
        "config": {
            "algorithm": algorithm,
            "factors": factors,
            "regularization": regularization,
            "iterations": iterations,
            "threshold": threshold,
        },
    }, model_path)
    print(f"\n[OK] Model saved: {model_path}")

    # ---------- 保存结果 ----------
    results = {
        "config": {
            "algorithm": algorithm,
            "factors": factors,
            "iterations": iterations,
            "threshold": threshold,
        },
        "metrics": {
            "precision@5": float(prec5),
            "recall@5": float(rec5),
            "precision@10": float(prec10),
            "recall@10": float(rec10),
            "precision@20": float(prec20),
            "recall@20": float(rec20),
            "auc": float(auc),
        },
    }

    result_path = os.path.join(RESULTS_DIR, f"results_{algorithm}_{factors}d.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Results saved: {result_path}")

    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练电影推荐模型")
    parser.add_argument("--algo", type=str, default="als",
                        choices=["als", "bpr", "lmf"],
                        help="推荐算法")
    parser.add_argument("--factors", type=int, default=64,
                        help="隐向量维度")
    parser.add_argument("--reg", type=float, default=0.01,
                        help="正则化系数")
    parser.add_argument("--iters", type=int, default=30,
                        help="训练迭代次数")
    args = parser.parse_args()

    model, results = train(
        algorithm=args.algo,
        factors=args.factors,
        regularization=args.reg,
        iterations=args.iters,
    )

    print(f"\nTraining Complete | "
          f"Prec@10: {results['metrics']['precision@10']:.4f} | "
          f"Recall@10: {results['metrics']['recall@10']:.4f}")
