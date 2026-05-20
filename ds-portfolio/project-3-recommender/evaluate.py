"""
模型评估模块
离线评估推荐系统的各项指标 (基于 implicit)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import (
    load_movielens_100k,
    build_implicit_feedback,
    train_test_split_interactions,
)
from train import precision_recall_at_k, auc_score_manual

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def compare_algorithms():
    """对比不同算法的效果"""
    print("\n" + "=" * 60)
    print("Algorithm Comparison Experiment")
    print("=" * 60)

    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking

    data = load_movielens_100k()
    interactions = build_implicit_feedback(data, threshold=4.0)
    train_mat, test_mat = train_test_split_interactions(interactions)

    algorithms = {
        "ALS": AlternatingLeastSquares(factors=64, regularization=0.01, iterations=30, random_state=42, num_threads=4),
        "BPR": BayesianPersonalizedRanking(factors=64, learning_rate=0.05, iterations=30, random_state=42, num_threads=4),
    }

    results = []
    for name, model in algorithms.items():
        print(f"\n[Experiment] {name} ...")
        model.fit(train_mat, show_progress=False)

        prec10, rec10 = precision_recall_at_k(model, train_mat, test_mat, k=10)
        auc = auc_score_manual(model, train_mat, test_mat)

        metrics = {"algorithm": name, "precision@10": float(prec10), "recall@10": float(rec10), "auc": float(auc)}
        results.append(metrics)
        print(f"  Prec@10: {prec10:.4f}  Recall@10: {rec10:.4f}  AUC: {auc:.4f}")

    # Save
    result_path = os.path.join(RESULTS_DIR, "algorithm_comparison.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Plot
    df_plot = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_plot))
    w = 0.25
    ax.bar(x - w, df_plot["precision@10"], w, label="Precision@10")
    ax.bar(x, df_plot["recall@10"], w, label="Recall@10")
    ax.bar(x + w, df_plot["auc"], w, label="AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["algorithm"])
    ax.set_title("Algorithm Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "algorithm_comparison.png"), dpi=150)
    plt.close()
    print(f"\n[OK] Chart saved")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommender model")
    parser.add_argument("--compare", action="store_true", help="Compare algorithms")
    args = parser.parse_args()

    if args.compare:
        compare_algorithms()
    else:
        # Default: load best model and evaluate
        print("\nLoading best model and evaluating...")
        from implicit.als import AlternatingLeastSquares

        data = load_movielens_100k()
        interactions = build_implicit_feedback(data, threshold=4.0)
        train_mat, test_mat = train_test_split_interactions(interactions)

        model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=30, random_state=42, num_threads=4)
        print("[Training] ...")
        model.fit(train_mat, show_progress=False)

        prec10, rec10 = precision_recall_at_k(model, train_mat, test_mat, k=10)
        prec5, rec5 = precision_recall_at_k(model, train_mat, test_mat, k=5)
        prec20, rec20 = precision_recall_at_k(model, train_mat, test_mat, k=20)
        auc = auc_score_manual(model, train_mat, test_mat)

        metrics = {
            "precision@5": float(prec5), "recall@5": float(rec5),
            "precision@10": float(prec10), "recall@10": float(rec10),
            "precision@20": float(prec20), "recall@20": float(rec20),
            "auc": float(auc),
        }
        print("\n[Final Evaluation Results]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        result_path = os.path.join(RESULTS_DIR, "final_evaluation.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[OK] Results saved: {result_path}")
