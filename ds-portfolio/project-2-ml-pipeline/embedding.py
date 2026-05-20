"""
sentence-transformers 文本 Embedding 特征提取器
作为 TF-IDF+SVD 的替代方案，提供现代语义特征。

首次运行自动下载模型 (~80MB)，之后本地缓存。
"""

import numpy as np


def get_embedding_model(name: str = "all-MiniLM-L6-v2"):
    """延迟加载 sentence-transformers 模型。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("请安装: pip install sentence-transformers")
    return SentenceTransformer(name)


class EmbeddingFeatureExtractor:
    """使用 sentence-transformers 将文本转为固定维度向量。"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = get_embedding_model(self.model_name)
        return self._model

    @property
    def n_components(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def fit_transform(self, texts: np.ndarray) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=True, batch_size=32)

    def transform(self, texts: np.ndarray) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, batch_size=32)

    def save(self, path: str):
        import joblib
        joblib.dump({"model_name": self.model_name}, path)

    @staticmethod
    def load(path: str):
        import joblib
        data = joblib.load(path)
        return EmbeddingFeatureExtractor(model_name=data["model_name"])
