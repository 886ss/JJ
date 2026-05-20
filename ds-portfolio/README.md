# 📊 数据科学项目集 (Data Science Portfolio)

> 准毕业生 / 实习面试 · 三个递进式数据科学项目

---

## 项目概览

| # | 项目 | 定位 | 难度 | 周期 |
|---|------|------|------|------|
| 1 | [📊 电商数据分析 Dashboard](project-1-dashboard/) | 数据分析与可视化 | ⭐⭐ | 2 周 |
| 2 | [🤖 端到端 ML Pipeline](project-2-ml-pipeline/) | 机器学习全流程 | ⭐⭐⭐⭐ | 4 周 |
| 3 | [🎬 电影推荐系统](project-3-recommender/) | 推荐算法与系统工程 | ⭐⭐⭐ | 3 周 |

## 项目之间的递进关系

```
项目一                  项目二                    项目三
数据分析              机器学习建模              推荐系统
  ↓                      ↓                        ↓
理解业务指标  ──→  从分析到建模  ──→  深入经典业务场景
Dashboard              Pipeline                Recommender
```

**面试时的故事线：**

> *"我先从数据分析入手（Dashboard），理解了业务指标和数据特征；然后用机器学习建模解决文本分类问题（Pipeline），掌握了从实验到部署的全流程；最后深入到推荐系统这个经典业务场景，理解了协同过滤和冷启动等核心问题。三个项目覆盖了数据科学家的核心工作流——分析、建模、推荐。"*

## 技术栈全景图

```
┌────────────────────────────────────────────────────┐
│                    数据科学核心技能                    │
├──────────────┬──────────────────┬──────────────────┤
│   数据处理    │    机器学习       │    系统工程       │
├──────────────┼──────────────────┼──────────────────┤
│ Pandas       │ scikit-learn     │ FastAPI          │
│ NumPy        │ implicit         │ Streamlit        │
│ Plotly       │ NumPy/SciPy      │ Docker           │
│ Matplotlib   │ MLflow           │ uvicorn          │
│ Seaborn      │ TF-IDF / SVD     │ REST API 设计     │
└──────────────┴──────────────────┴──────────────────┘
```

## 快速导航

### 📊 项目一：电商数据分析 Dashboard

```
cd project-1-dashboard
pip install -r requirements.txt
streamlit run app.py
```

→ [项目一 README](project-1-dashboard/README.md)

### 🤖 项目二：端到端 ML Pipeline

```
cd project-2-ml-pipeline
pip install -r requirements.txt
python train.py          # 训练
python evaluate.py        # 评估
uvicorn serve:app --reload  # 启动 API
```

→ [项目二 README](project-2-ml-pipeline/README.md)

### 🎬 项目三：电影推荐系统

```
cd project-3-recommender
pip install -r requirements.txt
python train.py           # 训练
uvicorn serve:app --reload --port 8001  # 启动 API
```

→ [项目三 README](project-3-recommender/README.md)

## 简历整合建议

将三个项目合并为一个项目集写在简历上：

> **数据科学项目集：分析 → 建模 → 推荐**（3 个月）  
> *技术栈：Python + Pandas + scikit-learn + implicit + Streamlit + FastAPI + MLflow + Plotly*  
> - 构建电商数据分析 Dashboard，清洗 12,000+ 条订单数据，集成销售趋势、RFM 客户分层等 8 个交互式组件  
> - 实现端到端文本分类 Pipeline，涵盖 TF-IDF 特征工程、多模型对比、MLflow 实验追踪、FastAPI 模型部署  
> - 开发混合推荐引擎，融合协同过滤与电影内容特征，Precision@10 达 0.38，对比 WARP/BPR/Logistic 三种损失函数

## 面试准备清单

- [ ] 每个项目都能用 2 分钟讲清楚：**业务问题 → 技术方案 → 量化结果**
- [ ] 能回答"为什么选这个技术方案而不是 XX"
- [ ] 项目代码已 push 到 GitHub，README 完善且有截图
- [ ] 每个项目都有 README 中的「简历亮点」可在面试前温习

## 目录结构

```
ds-portfolio/
├── README.md
├── project-1-dashboard/
│   ├── app.py               # Streamlit 主应用
│   ├── data_loader.py       # 数据生成 / 预处理
│   ├── visualizations.py    # Plotly 图表
│   ├── requirements.txt
│   └── README.md
├── project-2-ml-pipeline/
│   ├── config.py            # 全局配置
│   ├── data.py              # 数据加载
│   ├── features.py          # 特征工程
│   ├── train.py             # 训练
│   ├── tune.py              # 超参调优
│   ├── evaluate.py          # 评估
│   ├── predict.py           # CLI 预测
│   ├── serve.py             # FastAPI
│   ├── Dockerfile
│   ├── requirements.txt
│   └── README.md
└── project-3-recommender/
    ├── data_loader.py       # MovieLens 加载
    ├── train.py             # 模型训练
    ├── evaluate.py          # 评估与对比
    ├── serve.py             # FastAPI
    ├── requirements.txt
    └── README.md
```
