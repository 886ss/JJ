"""
工具函数模块
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

import config


def save_results(results: dict, filename: str = None):
    """保存评估结果为 JSON 文件"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"

    path = config.RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"[OK] 结果已保存: {path}")
    return path


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    labels: list = None) -> dict:
    """
    计算分类指标
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # 每个类别的指标
    per_class = {}
    if labels is not None:
        class_precision, class_recall, class_f1, class_support = (
            precision_recall_fscore_support(
                y_true, y_pred, labels=labels, zero_division=0
            )
        )
        for i, label in enumerate(labels):
            per_class[label] = {
                "precision": round(float(class_precision[i]), 4),
                "recall": round(float(class_recall[i]), 4),
                "f1": round(float(class_f1[i]), 4),
                "support": int(class_support[i]),
            }

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4),
        "per_class": per_class,
        "num_samples": len(y_true),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: list, save_path: str = None):
    """
    绘制并保存混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("预测标签")
    ax.set_ylabel("真实标签")
    ax.set_title("混淆矩阵")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[OK] 混淆矩阵已保存: {save_path}")
        plt.close(fig)
    else:
        save_path = config.RESULTS_DIR / "confusion_matrix.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                target_names: list):
    """打印分类报告"""
    report = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    print("\n" + "=" * 50)
    print("分类报告")
    print("=" * 50)
    print(report)
    print("=" * 50)
