"""
超参数调优模块
使用 GridSearchCV 进行网格搜索
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import config
from data import load_20newsgroups, preprocess_texts
from features import TextFeatureExtractor
from train import get_model
from utils import compute_metrics


def tune(model_name: str = "logistic", n_jobs: int = -1):
    """
    超参数网格搜索
    """
    # ---------- 加载数据 ----------
    print("\n" + "=" * 50)
    print(f"🔧 超参数调优: {model_name}")
    print("=" * 50)

    train_df, val_df, X_test_raw, y_test = load_20newsgroups()
    labels = sorted(train_df["label"].unique())

    X_train_text = preprocess_texts(train_df["text"].values)
    X_test_text = preprocess_texts(X_test_raw)

    y_train = train_df["label"].values

    # 特征提取
    extractor = TextFeatureExtractor()
    X_train = extractor.fit_transform(X_train_text)
    X_test = extractor.transform(X_test_text)

    # ---------- 参数网格 ----------
    param_grids = {
        "logistic": {
            "C": [0.1, 0.5, 1.0, 5.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        "svm": {
            "C": [0.1, 0.5, 1.0, 5.0, 10.0],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
        "rf": {
            "n_estimators": [100, 200, 500],
            "max_depth": [20, 50, None],
            "min_samples_split": [2, 5, 10],
        },
        "xgb": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.05, 0.1, 0.2],
        },
    }

    param_grid = param_grids.get(model_name, param_grids["logistic"])
    print(f"\n[参数] 搜索空间: {json.dumps(param_grid, indent=2)}")
    print(f"[参数] 组合数: {np.prod([len(v) for v in param_grid.values()])}")

    # ---------- 网格搜索 ----------
    base_model = get_model(model_name)
    cv = StratifiedKFold(n_splits=3, shuffle=True,
                         random_state=config.RANDOM_SEED)

    grid = GridSearchCV(
        base_model, param_grid,
        cv=cv, scoring="f1_weighted",
        n_jobs=n_jobs, verbose=1,
        return_train_score=True,
    )

    print("\n[搜索] 开始网格搜索...\n")
    grid.fit(X_train, y_train)

    # ---------- 最优模型评估 ----------
    best_model = grid.best_estimator_
    y_test_pred = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred, labels)

    print(f"\n[最优] 参数: {grid.best_params_}")
    print(f"[最优] 交叉验证 F1: {grid.best_score_:.4f}")
    print(f"[测试] 准确率: {test_metrics['accuracy']:.4f}  "
          f"F1: {test_metrics['f1_weighted']:.4f}")

    # ---------- 保存结果 ----------
    results = {
        "model": model_name,
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "test_metrics": test_metrics,
        "total_combinations": len(grid.cv_results_["params"]),
    }

    result_path = config.RESULTS_DIR / "tuning_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 调优结果已保存: {result_path}")

    return best_model, extractor, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="超参数调优")
    parser.add_argument("--model", type=str, default="logistic",
                        choices=["logistic", "svm", "rf", "xgb"])
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    model, extractor, results = tune(args.model, args.n_jobs)

    print("\n✅ 调优完成")
    print(f"   最优参数: {results['best_params']}")
    print(f"   最优 CV F1: {results['best_cv_score']:.4f}")
    print(f"   测试 F1: {results['test_metrics']['f1_weighted']:.4f}")
