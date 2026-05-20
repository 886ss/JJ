"""
模型评估模块
在测试集上输出完整的评估报告
"""

import argparse

import numpy as np
import joblib

import config
from data import load_20newsgroups, preprocess_texts, clean_text
from features import TextFeatureExtractor
from utils import (
    compute_metrics, plot_confusion_matrix,
    print_classification_report, save_results,
)


def evaluate(model_path: str = None, extractor_path: str = None):
    """
    加载模型并在测试集上评估
    """
    print("\n" + "=" * 50)
    print("📋 模型评估")
    print("=" * 50)

    # ---------- 加载数据 ----------
    _, _, X_test_raw, y_test = load_20newsgroups()
    labels = sorted(set(y_test))

    X_test_text = preprocess_texts(X_test_raw)

    # ---------- 加载特征提取器 ----------
    if extractor_path is None:
        extractor_path = config.MODEL_DIR / "feature_extractor.pkl"

    extractor = TextFeatureExtractor.load(extractor_path)
    X_test = extractor.transform(X_test_text)

    # ---------- 加载模型 ----------
    if model_path is None:
        # 自动检测
        candidates = sorted(config.MODEL_DIR.glob("*_model.pkl"))
        if not candidates:
            raise FileNotFoundError(
                f"未找到模型文件，请先运行 train.py\n"
                f"在 {config.MODEL_DIR} 中搜索失败"
            )
        model_path = candidates[-1]  # 选最新的
        print(f"[信息] 自动选择模型: {model_path.name}")

    model = joblib.load(model_path)
    print(f"[OK]  已加载模型: {model_path}")

    # ---------- 预测 + 评估 ----------
    y_pred = model.predict(X_test)

    print_classification_report(y_test, y_pred, labels)
    metrics = compute_metrics(y_test, y_pred, labels)

    print(f"\n[总体] 准确率: {metrics['accuracy']:.4f}")
    print(f"[总体] Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"[总体] Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"[总体] F1 (weighted): {metrics['f1_weighted']:.4f}")

    # 绘制混淆矩阵
    cm_path = config.RESULTS_DIR / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, labels, str(cm_path))

    # 保存评估报告
    report = {
        "model": str(model_path),
        "overall": {
            "accuracy": metrics["accuracy"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
            "num_samples": metrics["num_samples"],
        },
        "per_class": metrics["per_class"],
    }
    save_results(report, "evaluation_results.json")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估新闻分类模型")
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径（不指定则自动选择）")
    parser.add_argument("--extractor", type=str, default=None,
                        help="特征提取器路径")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        extractor_path=args.extractor,
    )

    print("\n✅ 评估完成")
