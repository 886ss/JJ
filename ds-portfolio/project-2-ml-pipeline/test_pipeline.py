"""pytest for ML Pipeline 文本分类"""
import numpy as np
import pytest
from data import clean_text, load_20newsgroups
from features import TextFeatureExtractor


class TestDataLoading:
    def test_loads_data(self):
        train_df, val_df, X_test, y_test = load_20newsgroups()
        assert len(train_df) > 10000
        assert len(val_df) > 1000
        assert len(X_test) > 3000
        assert len(y_test) == len(X_test)

    def test_six_classes(self):
        train_df, _, _, _ = load_20newsgroups()
        labels = train_df["label"].unique()
        assert len(labels) == 6  # 计算机/科学/社科/休闲/哲学/其他

    def test_no_empty_text(self):
        train_df, val_df, X_test, _ = load_20newsgroups()
        assert (train_df["text"].str.len() > 50).all()
        assert (val_df["text"].str.len() > 50).all()
        assert all(len(t) > 50 for t in X_test)


class TestTextCleaning:
    def test_lowercase(self):
        result = clean_text("Hello WORLD")
        assert result == "hello world"

    def test_removes_urls(self):
        result = clean_text("visit https://example.com now")
        assert "https" not in result
        assert "example" not in result

    def test_removes_numbers(self):
        result = clean_text("call 123 4567 now")
        for digit in "0123456789":
            assert digit not in result

    def test_removes_emails(self):
        result = clean_text("email test@example.com now")
        assert "@" not in result

    def test_strips_whitespace(self):
        result = clean_text("  hello  world  ")
        assert result == "hello world"


class TestFeatureExtractor:
    def test_fit_transform_shape(self):
        texts = np.array([
            "the quick brown fox jumps over the lazy dog",
            "machine learning is a field of artificial intelligence",
            "deep learning models are used for nlp tasks",
            "python is a popular programming language for data science",
            "neural networks are the foundation of modern ai systems",
            "natural language processing is a subfield of machine learning",
        ])
        extractor = TextFeatureExtractor(max_features=100, n_components=5)
        X = extractor.fit_transform(texts)
        assert X.shape == (6, 5)

    def test_transform_uses_fit_vocab(self):
        texts_train = np.array([
            "hello world", "hello python", "world news", "python code"
        ])
        texts_test = np.array(["hello again"])
        extractor = TextFeatureExtractor(max_features=50, n_components=3)
        extractor.fit(texts_train)
        X_test = extractor.transform(texts_test)
        assert X_test.shape == (1, 3)
