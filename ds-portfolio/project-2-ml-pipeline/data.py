"""
数据加载与预处理模块
"""

import re
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

import config


def load_20newsgroups() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    加载 20 Newsgroups 数据集，映射为 4 大类，拆分训练/测试/验证集

    返回:
        train_df, val_df (pd.DataFrame): 训练集 / 验证集（含 text + label）
        X_test_raw, y_test (np.ndarray): 测试集原文 + 标签
    """
    # ---------- 加载完整数据 ----------
    newsgroups = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        random_state=config.RANDOM_SEED,
    )

    # 映射目标名 → 大类
    target_names = newsgroups["target_names"]
    large_categories = [config.CATEGORY_MAP.get(name, "其他") for name in target_names]

    df = pd.DataFrame({
        "text": newsgroups["data"],
        "target": newsgroups["target"],
        "label": [large_categories[t] for t in newsgroups["target"]],
    })

    # 移除空文本
    df = df[df["text"].str.strip().str.len() > 50].reset_index(drop=True)

    print(f"[数据] 总样本数: {len(df)}")
    print(f"[数据] 类别分布:\n{df['label'].value_counts()}")

    # ---------- 分层拆分 ----------
    # 先拆出测试集（20%）
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=config.TEST_SIZE,
        stratify=df["label"],
        random_state=config.RANDOM_SEED,
    )

    # 再从训练集中拆出验证集（10% → 总体的 8%）
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=config.VAL_SIZE,
        stratify=df.iloc[train_idx]["label"],
        random_state=config.RANDOM_SEED,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_data = df.iloc[test_idx]

    print(f"[数据] 训练集: {len(train_df)} | 验证集: {len(val_df)} "
          f"| 测试集: {len(test_data)}")

    return train_df, val_df, test_data["text"].values, test_data["label"].values


def clean_text(text: str) -> str:
    """
    文本清洗：
    1. 转小写
    2. 移除 URL、邮箱、数字
    3. 移除多余空格
    """
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_texts(texts: np.ndarray) -> np.ndarray:
    """批量清洗文本"""
    return np.array([clean_text(t) for t in texts])
