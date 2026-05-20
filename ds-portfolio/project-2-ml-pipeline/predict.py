"""
命令行预测模块
"""

import argparse

import numpy as np
import joblib

import config
from data import clean_text
from features import TextFeatureExtractor


def load_artifacts(model_path: str = None,
                   extractor_path: str = None):
    """加载模型和特征提取器"""
    if extractor_path is None:
        extractor_path = config.MODEL_DIR / "feature_extractor.pkl"
    if model_path is None:
        candidates = sorted(config.MODEL_DIR.glob("*_model.pkl"))
        if not candidates:
            raise FileNotFoundError("未找到模型文件，请先运行 train.py")
        model_path = candidates[-1]

    extractor = TextFeatureExtractor.load(extractor_path)
    model = joblib.load(model_path)
    return model, extractor


def predict_single(text: str, model, extractor) -> dict:
    """预测单条文本"""
    cleaned = clean_text(text)
    X = extractor.transform(np.array([cleaned]))
    pred = model.predict(X)[0]

    probs = None
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(X)[0]
        classes = model.classes_
        probs = {cls: round(float(p), 4) for cls, p in zip(classes, proba_arr)}

    return {"text": text[:200] + ("..." if len(text) > 200 else ""),
            "prediction": str(pred),
            "probabilities": probs}


def batch_predict(texts: list, model, extractor) -> list:
    """批量预测"""
    cleaned = [clean_text(t) for t in texts]
    X = extractor.transform(np.array(cleaned))
    preds = model.predict(X)
    return list(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="新闻分类预测")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--text", type=str, default=None,
                        help="待分类文本")
    parser.add_argument("--file", type=str, default=None,
                        help="包含待分类文本的文件（每行一条）")
    args = parser.parse_args()

    model, extractor = load_artifacts(args.model)

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        preds = batch_predict(lines, model, extractor)
        for text, pred in zip(lines, preds):
            print(f"[{pred}] {text[:80]}...")
    elif args.text:
        result = predict_single(args.text, model, extractor)
        print(f"\n📰 文本: {result['text']}")
        print(f"🏷️  分类: {result['prediction']}")
        if result["probabilities"]:
            print("🎯 概率:")
            for cls, prob in sorted(result["probabilities"].items(),
                                    key=lambda x: -x[1]):
                bar = "█" * int(prob * 30)
                print(f"  {cls:6s}  {prob:.2%} {bar}")
    else:
        # 交互模式
        print("\n📰 新闻分类器 (输入 'quit' 退出)\n")
        while True:
            text = input("请输入英文新闻文本: ").strip()
            if text.lower() == "quit":
                break
            if not text:
                continue
            result = predict_single(text, model, extractor)
            print(f"  → {result['prediction']}")
            if result["probabilities"]:
                for cls, prob in sorted(result["probabilities"].items(),
                                        key=lambda x: -x[1]):
                    print(f"    {cls}: {prob:.2%}")
            print()
