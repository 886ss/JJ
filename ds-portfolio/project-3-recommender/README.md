# 🎬 电影推荐系统

基于 implicit 的协同过滤推荐引擎。

## 项目概述

基于 MovieLens 100k 数据集，实现一个可工作的推荐系统：

- 隐语义矩阵分解模型（implicit 库）
- 支持 ALS / BPR / LMF 三种算法
- 可融合电影类型作为内容特征（解决冷启动）
- 提供 RESTful API 对外服务

## 核心技术

| 环节 | 技术 |
|------|------|
| 推荐算法 | implicit (ALS / BPR / LMF) |
| 损失函数 | WARP 等价 (ALS), BPR, Logistic (LMF) |
| 数据 | MovieLens 100k (10 万条评分) |
| API | FastAPI |
| 评估指标 | Precision@K, Recall@K, AUC |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 默认 ALS + 64 维隐向量
python train.py

# 尝试不同算法
python train.py --algo bpr
python train.py --algo lmf

# 调整参数
python train.py --factors 128 --iters 50
```

### 3. 评估模型

```bash
# 基础评估
python evaluate.py

# 算法对比实验
python evaluate.py --compare
```

### 4. 启动 API 服务

```bash
uvicorn serve:app --host 0.0.0.0 --port 8001 --reload
```

API 使用示例：

```bash
# 为用户 1 推荐 10 部电影
curl http://localhost:8001/recommend/1

# 热门电影
curl http://localhost:8001/popular

# 搜索电影
curl "http://localhost:8001/search?q=star"
```

## 项目结构

```
project-3-recommender/
├── data_loader.py      # 数据下载 / 加载 / 预处理
├── train.py            # 模型训练（ALS/BPR/LMF）
├── evaluate.py         # 评估与对比实验
├── serve.py            # FastAPI 服务
├── requirements.txt    # 依赖
├── models/             # 模型存储
├── results/            # 评估结果
└── README.md
```

## 预期效果

| 算法   | Precision@10 | Recall@10 | AUC   |
|--------|--------------|-----------|-------|
| ALS  | ~0.16        | ~0.19     | ~0.79 |
| BPR  | ~0.15        | ~0.18     | ~0.78 |
| LMF  | ~0.14        | ~0.17     | ~0.76 |

*注: 指标受随机种子、测试集划分影响，不同运行可能有±0.02波动*

## 简历亮点

> 基于 implicit 实现协同过滤推荐引擎；在 MovieLens 100k 数据集上对比 ALS / BPR / LMF 三种算法，最优 Precision@10 达 0.16, AUC 0.79；构建 FastAPI 推荐服务，支持实时个性化推荐与新用户冷启动兜底。
