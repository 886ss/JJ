"""
特征工程模块
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import joblib

import config


class TextFeatureExtractor:
    """
    文本特征提取器
    管道：清洗文本 → TF-IDF → SVD 降维 → 标准化
    """

    def __init__(self, max_features: int = config.MAX_FEATURES,
                 ngram_range: tuple = config.NGRAM_RANGE,
                 n_components: int = 100):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.n_components = n_components

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.svd = TruncatedSVD(n_components=n_components,
                                random_state=config.RANDOM_SEED)
        self.scaler = StandardScaler(with_mean=False)

    def fit(self, texts: np.ndarray) -> "TextFeatureExtractor":
        print(f"[特征] 原始文本数: {len(texts)}")
        # TF-IDF
        X_tfidf = self.vectorizer.fit_transform(texts)
        print(f"[特征] TF-IDF 维度: {X_tfidf.shape[1]}")

        # SVD 降维
        X_svd = self.svd.fit_transform(X_tfidf)
        explained = self.svd.explained_variance_ratio_.sum()
        print(f"[特征] SVD → {self.n_components} 维 | "
              f"累计方差解释率: {explained:.2%}")

        # 标准化
        self.scaler.fit(X_svd)

        return self

    def transform(self, texts: np.ndarray) -> np.ndarray:
        X_tfidf = self.vectorizer.transform(texts)
        X_svd = self.svd.transform(X_tfidf)
        return self.scaler.transform(X_svd)

    def fit_transform(self, texts: np.ndarray) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str):
        joblib.dump(self, path)
        print(f"[OK] 特征提取器已保存: {path}")

    @classmethod
    def load(cls, path: str) -> "TextFeatureExtractor":
        return joblib.load(path)
