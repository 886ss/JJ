"""
模型训练模块
支持：逻辑回归、线性 SVM、随机森林
集成 MLflow 实验追踪
"""

import argparse
import json

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib

import config
from data import load_20newsgroups, preprocess_texts, clean_text
from features import TextFeatureExtractor
from utils import compute_metrics


def get_model(model_name: str):
    """获取模型实例"""
    models = {
        "logistic": LogisticRegression(
            C=1.0, max_iter=2000, random_state=config.RANDOM_SEED,
            multi_class="multinomial", n_jobs=-1,
        ),
        "svm": LinearSVC(
            C=1.0, max_iter=2000, random_state=config.RANDOM_SEED,
            dual=False,
        ),
        "rf": RandomForestClassifier(
            n_estimators=200, max_depth=50, min_samples_split=5,
            random_state=config.RANDOM_SEED, n_jobs=-1,
        ),
    }
    return models.get(model_name, models["logistic"])


def train(model_name: str, tune_params: dict = None):
    """
    完整训练流程：
    1. 加载数据
    2. 文本清洗
    3. 特征工程
    4. 训练模型
    5. MLflow 追踪
    """
    # ---------- 加载数据 ----------
    print("\n" + "=" * 50)
    print(f"🚀 开始训练: {model_name}")
    print("=" * 50)

    train_df, val_df, X_test_raw, y_test = load_20newsgroups()

    labels = sorted(train_df["label"].unique())
    print(f"\n[信息] 类别数: {len(labels)} → {labels}")

    # ---------- 文本清洗 ----------
    print("\n[清洗] 正在清洗文本...")
    X_train_text = preprocess_texts(train_df["text"].values)
    X_val_text = preprocess_texts(val_df["text"].values)
    X_test_text = preprocess_texts(X_test_raw)

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    # ---------- 特征提取 ----------
    print("\n[特征] 正在提取特征...")
    extractor = TextFeatureExtractor()
    X_train = extractor.fit_transform(X_train_text)
    X_val = extractor.transform(X_val_text)
    X_test = extractor.transform(X_test_text)

    # ---------- 训练模型 ----------
    print(f"\n[训练] 训练 {model_name.upper()} 模型...")
    params = tune_params or {}
    model = get_model(model_name)
    model.set_params(**params)

    # 交叉验证
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=config.CV_FOLDS, scoring="f1_weighted")
    print(f"[CV]   {config.CV_FOLDS}-Fold CV F1: "
          f"{cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    # 全量训练
    model.fit(X_train, y_train)

    # 验证集评估
    y_val_pred = model.predict(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred, labels)

    print(f"[验证] 准确率: {val_metrics['accuracy']:.4f}  "
          f"F1: {val_metrics['f1_weighted']:.4f}")

    # 测试集评估
    y_test_pred = model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred, labels)

    print(f"[测试] 准确率: {test_metrics['accuracy']:.4f}  "
          f"F1: {test_metrics['f1_weighted']:.4f}")

    # ---------- MLflow 追踪 ----------
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{model_name}_{len(labels)}class") as run:
        mlflow.log_param("model", model_name)
        mlflow.log_param("max_features", config.MAX_FEATURES)
        mlflow.log_param("ngram_range", str(config.NGRAM_RANGE))
        mlflow.log_param("n_components", extractor.n_components)
        mlflow.log_params(params)

        mlflow.log_metric("cv_f1_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_f1_std", float(cv_scores.std()))
        mlflow.log_metric("val_accuracy", val_metrics["accuracy"])
        mlflow.log_metric("val_f1", val_metrics["f1_weighted"])
        mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
        mlflow.log_metric("test_f1", test_metrics["f1_weighted"])

        # 保存模型 & 特征提取器
        mlflow.sklearn.log_model(model, "model")
        extractor_path = config.MODEL_DIR / "feature_extractor.pkl"
        extractor.save(str(extractor_path))
        mlflow.log_artifact(str(extractor_path))

        model_path = config.MODEL_DIR / f"{model_name}_model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        print(f"\n[MLflow] Run ID: {run.info.run_id}")
        print(f"[MLflow] 查看实验: mlflow ui --backend-store-uri "
              f"{config.MLFLOW_TRACKING_URI}")

    # ---------- 保存最终结果 ----------
    final_results = {
        "model": model_name,
        "cv_scores": {"mean": float(cv_scores.mean()),
                      "std": float(cv_scores.std()),
                      "folds": cv_scores.tolist()},
        "validation": val_metrics,
        "test": test_metrics,
        "labels": labels,
        "run_id": run.info.run_id,
    }

    result_path = config.RESULTS_DIR / "training_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"[OK] 训练结果已保存: {result_path}")

    return model, extractor, final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练新闻分类模型")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        choices=["logistic", "svm", "rf"],
                        help="模型类型")
    parser.add_argument("--C", type=float, default=1.0,
                        help="正则化强度（logistic/svm）")
    args = parser.parse_args()

    params = {"C": args.C} if args.model in ("logistic", "svm") else {}

    model, extractor, results = train(args.model, tune_params=params)

    print("\n" + "=" * 50)
    print("✅ 训练完成")
    print(f"   模型: {args.model}")
    print(f"   测试集 F1: {results['test']['f1_weighted']:.4f}")
    print(f"   测试集准确率: {results['test']['accuracy']:.4f}")
    print("=" * 50)
