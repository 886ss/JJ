"""
全局配置模块
"""

import os
from pathlib import Path

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ---------- 数据集 ----------
# 使用 scikit-learn 自带的 20 Newsgroups 数据集
# 将 20 个新闻组归类为 4 个大类
CATEGORY_MAP = {
    # 计算机
    "comp.graphics": "计算机",
    "comp.os.ms-windows.misc": "计算机",
    "comp.sys.ibm.pc.hardware": "计算机",
    "comp.sys.mac.hardware": "计算机",
    "comp.windows.x": "计算机",
    # 科学
    "sci.crypt": "科学",
    "sci.electronics": "科学",
    "sci.med": "科学",
    "sci.space": "科学",
    # 社科
    "talk.politics.misc": "社科",
    "talk.politics.guns": "社科",
    "talk.politics.mideast": "社科",
    # 休闲
    "rec.autos": "休闲",
    "rec.motorcycles": "休闲",
    "rec.sport.baseball": "休闲",
    "rec.sport.hockey": "休闲",
    # 宗教哲学
    "alt.atheism": "哲学",
    "soc.religion.christian": "哲学",
    "talk.religion.misc": "哲学",
}

# ---------- 模型配置 ----------
RANDOM_SEED = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# TF-IDF 特征
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)

# ---------- MLflow ----------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
MLFLOW_EXPERIMENT = "news-classification"

# ---------- 训练 ----------
DEFAULT_MODEL = "logistic"  # 可选: logistic, svm, rf
CV_FOLDS = 5
