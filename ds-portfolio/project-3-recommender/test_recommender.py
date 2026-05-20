"""pytest for 推荐系统"""
import numpy as np
from scipy.sparse import csr_matrix
from data_loader import build_implicit_feedback, train_test_split_interactions


def _make_dummy_interactions():
    """构造一个小型交互矩阵用于测试"""
    data = np.array([5.0, 4.0, 3.0, 5.0, 2.0, 4.0, 1.0, 5.0], dtype=np.float64)
    row = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    col = np.array([0, 2, 1, 3, 0, 2, 1, 3])
    return csr_matrix((data, (row, col)), shape=(4, 4))


class TestImplicitFeedback:
    def test_threshold_filters(self):
        interactions = _make_dummy_interactions()
        imp = build_implicit_feedback(
            {"interactions": interactions}, threshold=4.0
        )
        # 评分>=4的: 5.0,4.0,5.0,4.0,5.0 = 5个正样本
        assert imp.nnz == 5

    def test_all_binary(self):
        interactions = _make_dummy_interactions()
        imp = build_implicit_feedback(
            {"interactions": interactions}, threshold=4.0
        )
        assert (imp.data == 1.0).all()

    def test_threshold_3(self):
        interactions = _make_dummy_interactions()
        imp = build_implicit_feedback(
            {"interactions": interactions}, threshold=3.0
        )
        # 评分>=3: 5.0,4.0,3.0,5.0,4.0,5.0 = 6正样本
        assert imp.nnz == 6


class TestTrainTestSplit:
    def test_shapes_match(self):
        interactions = _make_dummy_interactions()
        train, test = train_test_split_interactions(
            interactions, test_percentage=0.25, random_state=42
        )
        assert train.shape == interactions.shape
        assert test.shape == interactions.shape

    def test_no_overlap(self):
        interactions = _make_dummy_interactions()
        train, test = train_test_split_interactions(
            interactions, test_percentage=0.3, random_state=42
        )
        overlap = train.multiply(test).nnz
        assert overlap == 0

    def test_split_ratio(self):
        interactions = _make_dummy_interactions()
        train, test = train_test_split_interactions(
            interactions, test_percentage=0.25, random_state=42
        )
        ratio = test.nnz / interactions.nnz
        assert 0.1 <= ratio <= 0.5
