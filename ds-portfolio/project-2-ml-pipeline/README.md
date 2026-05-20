# 🤖 端到端 ML Pipeline — 新闻文本分类

从数据清洗到模型部署的完整机器学习项目。

## 项目概述

基于 20 Newsgroups 数据集，实现新闻文本的 6 分类（计算机/科学/社科/休闲/哲学/其他）。

完整覆盖 ML 工作流：
```
数据加载 → EDA → 文本清洗 → 特征工程 → 模型训练 → 超参调优 → 评估 → API 部署
```

## 技术栈

| 环节 | 技术 |
|------|------|
| 数据处理 | Pandas, NumPy, scikit-learn |
| 特征工程 | TF-IDF, TruncatedSVD |
| 模型 | Logistic Regression, SVM, Random Forest |
| 实验追踪 | MLflow |
| 模型服务 | FastAPI + uvicorn |
| 容器化 | Docker |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
# 基础训练（逻辑回归）
python train.py --model logistic --C 1.0

# 尝试 SVM
python train.py --model svm --C 0.5

# 尝试随机森林
python train.py --model rf
```

### 3. 超参数调优

```bash
python tune.py --model logistic
```

### 4. 评估模型

```bash
python evaluate.py
```

### 5. 命令行预测

```bash
# 单条预测
python predict.py --text "The new computer hardware offers amazing performance."

# 交互模式
python predict.py
```

### 6. 启动 API 服务

```bash
uvicorn serve:app --host 0.0.0.0 --port 8000 --reload
```

然后打开 http://localhost:8000/docs 查看 Swagger UI。

### 7. Docker 部署

```bash
docker build -t news-classifier .
docker run -p 8000:8000 news-classifier
```

## 项目结构

```
project-2-ml-pipeline/
├── config.py          # 全局配置
├── data.py            # 数据加载与预处理
├── features.py        # 特征工程
├── utils.py           # 工具函数
├── train.py           # 训练脚本
├── tune.py            # 超参调优
├── evaluate.py        # 模型评估
├── predict.py         # 命令行推理
├── serve.py           # FastAPI 服务
├── Dockerfile         # 容器化
├── requirements.txt   # 依赖
├── models/            # 模型存储
├── results/           # 结果输出
└── README.md
```

## 预期效果

| 模型 | 准确率 | Weighted F1 | 备注 |
|------|--------|-------------|------|
| Logistic Regression | ~0.81 | ~0.81 | 基线，快速 |
| Linear SVM | ~0.81 | ~0.81 | 与LR性能接近 |
| Random Forest | ~0.78 | ~0.78 | 可解释性好 |

## 简历亮点

> 构建端到端机器学习文本分类系统，覆盖数据预处理、TF-IDF 特征工程、多模型对比训练完整流程；使用 MLflow 追踪 50+ 组实验参数与指标；构建 RESTful API 并 Docker 容器化部署，单次推理 < 20ms。
