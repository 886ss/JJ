# DS-Portfolio: Data Science Practicum

## Project Overview

This portfolio demonstrates the end-to-end data science workflow through three progressively deeper projects — from exploratory analysis to machine learning engineering to recommendation systems. Each project addresses a concrete business problem and delivers measurable results.

The projects are designed to answer three questions that mirror real-world data science responsibilities:

- **Can we turn raw transaction data into actionable business insights?** (Project 1)
- **Can we build, evaluate, and deploy a reliable text classification system?** (Project 2)
- **Can we deliver personalized recommendations at scale with measurable quality?** (Project 3)

All three projects include automated tests (28 test cases), RESTful API serving, interactive dashboards, and experiment tracking — reflecting engineering practices expected in a production team.

---

## Objectives

- **User behavior analysis** — Understand purchase patterns, time-based trends, and customer preferences from transaction logs
- **Customer value segmentation** — Quantify customer lifetime value using RFM modeling to distinguish high-value from at-risk users
- **Text classification automation** — Build a pipeline that classifies unstructured text into predefined categories with measurable accuracy
- **Personalized recommendation** — Implement collaborative filtering to surface relevant items for each user, with cold-start fallbacks
- **Reproducible experimentation** — Track every model run, parameter set, and evaluation metric for audit and comparison
- **Deployable artifacts** — Package models as REST APIs and interactive dashboards, not just notebooks

---

## Tech Stack

| Category | Tools |
| --- | --- |
| Data Processing | Python, Pandas, NumPy, SciPy |
| Machine Learning | scikit-learn, XGBoost, implicit |
| Feature Engineering | TF-IDF, TruncatedSVD, sentence-transformers |
| Experiment Tracking | MLflow |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web Framework | Streamlit, FastAPI, uvicorn |
| Containerization | Docker |
| RAG / LLM | LlamaIndex, Chroma, DeepSeek API |
| Testing | pytest (28 test cases) |

---

## Project Workflow

```text
                    ┌──────────────────────────┐
                    │     Data Ingestion        │
                    │  (CSV / API / Download)   │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │    Data Validation        │
                    │  (Missing / Distribution  │
                    │   / Outlier Detection)    │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │  Exploratory Analysis     │
                    │  (Trends / Distributions  │
                    │   / Correlations)         │
                    └────────────┬─────────────┘
                                 ↓
                    ┌──────────────────────────┐
                    │   Feature Engineering     │
                    │  (TF-IDF / SVD / Implicit │
                    │   Feedback Encoding)      │
                    └────────────┬─────────────┘
                                 ↓
               ┌─────────────────┴─────────────────┐
               ↓                                   ↓
    ┌──────────────────┐                ┌──────────────────┐
    │  Model Training   │                │  RFM / Segment   │
    │  + Hyperparam     │                │  Analysis        │
    │  Tuning + MLflow  │                │                  │
    └────────┬──────────┘                └────────┬─────────┘
             ↓                                    ↓
    ┌──────────────────┐                ┌──────────────────┐
    │  Evaluation      │                │  KPI Calculation  │
    │  (CV / P@K / AUC)│                │                  │
    └────────┬──────────┘                └────────┬─────────┘
             ↓                                    ↓
             └─────────────────┬──────────────────┘
                               ↓
                  ┌──────────────────────────┐
                  │   Deployment              │
                  │  (FastAPI / Streamlit /   │
                  │   Docker)                 │
                  └──────────────────────────┘
```

---

## Core Features

### Project 1 — E-Commerce Analytics Dashboard

**Business problem:** An e-commerce operator needs to understand sales performance, product mix, and customer quality at a glance — without running SQL queries or exporting to Excel.

- **5 real-time KPI cards** — Total revenue, order count, unique customers, average order value, refund rate. All respond to sidebar filters (date range, category, payment method).
- **8 interactive Plotly visualizations** — Daily trend (dual-axis), category ranking, payment distribution, review histogram, geographic breakdown, weekday-hour heatmap, monthly growth rate, RFM segment composition.
- **RFM customer segmentation** — Recency / Frequency / Monetary quartile scoring assigns every customer to one of 4 tiers (high-value, loyal, at-risk, lost). Displayed as an interactive donut chart with drill-down pivots.
- **Cross-tabulation explorer** — Dynamically pivot any two dimensions (category × payment, state × segment) for ad-hoc analysis.

### Project 2 — Text Classification ML Pipeline

**Business problem:** Manually categorizing thousands of incoming documents (news articles, support tickets, etc.) is slow and inconsistent. The goal is an automated classifier with known accuracy that can be deployed behind an API.

- **End-to-end training pipeline** — `config → data → features → train → tune → evaluate → serve`. Each stage is a standalone module with a clear interface.
- **Multi-model comparison** — Logistic Regression (baseline), Linear SVM, Random Forest, XGBoost trained under identical conditions. 5-fold stratified cross-validation ensures reliable performance estimates.
- **Feature extraction with dimensionality reduction** — TF-IDF (5000 features, n-gram 1–2) + TruncatedSVD (100 components) + StandardScaler. The extractor is serialized alongside the model for guaranteed consistency between training and inference.
- **MLflow experiment tracking** — Parameters, metrics, and artifacts logged for every run. Enables reproducibility and experiment comparison across model types.
- **Three serving interfaces** — CLI tool (`predict.py`) for batch prediction, FastAPI (`serve.py`) with Swagger docs, and Streamlit UI (`app.py`) with real-time news fetching for demo.
- **Docker deployment** — Single-container deployment with model pre-trained at build time.

### Project 3 — Movie Recommendation Engine

**Business problem:** In content platforms, 80% of consumption comes from recommendations. A quality recommender directly drives engagement and retention. This project builds one from scratch and measures its quality.

- **Three collaborative filtering algorithms** — ALS (Alternating Least Squares), BPR (Bayesian Personalized Ranking), LMF (Logistic Matrix Factorization) implemented via the `implicit` library. Each is trained on implicit feedback (rating ≥ 4 → positive).
- **Hybrid recommendation** — Collaborative filtering signals are combined with content-based features (18-dim movie genre vectors) via weighted scoring, providing reasonable recommendations for items with sparse interaction history.
- **Multi-endpoint REST API** — `/recommend/{user_id}` (personalized), `/recommend/{user_id}/hybrid` (CF + content), `/popular` (global hot list), `/search?q=` (keyword), `/similar/{item_id}` (content-based similarity).
- **Interactive exploration UI** — 5-tab Streamlit app covering personalized recommendations, popularity leaderboard, movie search, data analytics, and trending recent movies (lazy-loaded from GroupLens latest dataset).

### RAG Knowledge Base

- Codebase-aware Q&A system built with LlamaIndex + Chroma vector store + DeepSeek LLM.
- Ingests all project source files (`.py`, `.md`, `.json`, `.yml`) for documentation-free project understanding.
- Chunk size 1024 tokens, overlap 128; embeddings via `BAAI/bge-small-zh-v1.5`.

---

## Results

| Metric | Value |
| --- | --- |
| E-commerce orders processed | 12,000 |
| Customer segments identified (RFM) | 4 tiers |
| Dashboard interactive components | 8 charts + 5 KPIs |
| Text classification articles | 17,886 (6 categories) |
| Best classifier test F1 (Logistic Regression) | 0.812 |
| CV F1 (5-fold) | 0.806 ± 0.010 |
| Models compared (ML pipeline) | 4 (LR / SVM / RF / XGBoost) |
| MLflow tracked experiments | 4 full-pipeline runs |
| Recommendation ratings processed | 100,000 |
| Users × Items (MovieLens 100k) | 943 × 1,682 |
| Interaction matrix sparsity | 93.7% |
| Best recommender Precision@10 | 0.16 |
| Best recommender Recall@10 | 0.19 |
| Best recommender AUC | 0.79 |
| Recommendation algorithms compared | 3 (ALS / BPR / LMF) |
| API endpoints (total across projects) | 10+ |
| Automated tests | 28 (all passing) |

---

## Challenges & Solutions

### 1. Synthetic Data Distribution Bias

**Problem:** The initial date generation used `np.random.exponential`, causing January to receive 3,518 orders while December received only 279 (30:1 ratio). Monthly revenue growth rate was artificially inflated to 138%. Order timestamps lacked hour components, rendering the weekday-hour heatmap useless.

**Solution:** Replaced exponential sampling with uniform distribution. Added randomized hour components to timestamps. Post-fix: monthly order max/min ratio dropped to 1.14 (std=38), revenue MoM stabilized at ~1.2%, and heatmaps showed meaningful 24-hour patterns.

### 2. Recommendation Evaluation Bug

**Problem:** `filter_already_liked_items` was set to `False` in the recommendation generation step. Items the user had already interacted with (training data) dominated the top-K ranking, crowding out test-set items. Precision@10 was deflated to 0.016 — a 10× underestimation of true model quality.

**Solution:** Enabled `filter_already_liked_items=True`. Precision@10 corrected from 0.016 → 0.16, Recall@10 from 0.044 → 0.19. The fix was verified by re-running evaluation and confirming that recommended items were disjoint from the user's training interactions.

### 3. SVD Dimensionality Trade-off

**Problem:** TruncatedSVD with 100 components retained only 14.27% of TF-IDF variance. Increasing components would improve fidelity but also increase model size and inference latency — relevant for deployment behind an API.

**Solution:** Accepted the 100-dim trade-off given the deployment context. The Logistic Regression model with 100-dim SVD still achieved 81.2% test F1 — adequate for a 6-class text classifier. The design decision was documented so a future maintainer can increase dimensionality if higher accuracy is needed.

### 4. Label-Configuration Drift

**Problem:** The config file claimed "4 categories," while the actual category mapping defined 6 (including a catch-all "其他" class for `misc.forsale`). The README stated 5. This inconsistency would confuse anyone inheriting the project.

**Solution:** Corrected config comments and README to consistently document 6 categories. The "其他" class (933 samples, 5.2% of data) was left as-is since it represents a legitimate catch-all category for out-of-domain text.

---

## Highlights

- **Multi-project narrative arc** — Analysis → Modeling → Recommendation. Each project builds on skills from the previous one, forming a coherent story for interviews.
- **Engineering, not notebooks** — Modular Python packages with separated concerns, CLI entry points, and automated tests. Every model is deployable behind an API.
- **Experiment reproducibility** — MLflow tracks parameters, metrics, and artifacts. Model + feature extractor are serialized together so inference output matches training-time expectations.
- **Bug discovery and fix documentation** — Data distribution bias, evaluation metric deflation, and label-config mismatch were systematically identified through a structured DS review process and fixed with verified results.
- **RAG-powered project understanding** — The codebase itself is indexed into a Chroma vector store, enabling natural-language queries about implementation details.
- **Production-adjacent practices** — Docker containerization, Pydantic request validation, global error handling in FastAPI, Streamlit theme customization per project.

---

## Repository Structure

```text
ds-portfolio/
├── README.md                         # This file
│
├── project-1-dashboard/              # E-commerce Analytics
│   ├── app.py                        # Streamlit dashboard
│   ├── data_loader.py                # Data generation & RFM
│   ├── visualizations.py             # 8 Plotly chart functions
│   └── test_dashboard.py             # 12 tests
│
├── project-2-ml-pipeline/            # Text Classification
│   ├── config.py                     # Central configuration
│   ├── data.py                       # Data loading & cleaning
│   ├── features.py                   # TF-IDF + SVD extractors
│   ├── embedding.py                  # Sentence-transformer alternative
│   ├── train.py                      # Training with MLflow
│   ├── tune.py                       # GridSearchCV hyperparameter search
│   ├── evaluate.py                   # Evaluation & metrics
│   ├── predict.py                    # CLI inference
│   ├── serve.py                      # FastAPI server
│   ├── app.py                        # Streamlit UI
│   ├── Dockerfile                    # Container deployment
│   └── test_pipeline.py              # 11 tests
│
├── project-3-recommender/            # Movie Recommendations
│   ├── data_loader.py                # MovieLens ingestion
│   ├── train.py                      # ALS / BPR / LMF training
│   ├── evaluate.py                   # Evaluation & comparison
│   ├── serve.py                      # FastAPI recommendation API
│   ├── app.py                        # Streamlit UI
│   └── test_recommender.py           # 6 tests
│
└── rag/                              # RAG Knowledge Base
    ├── loader.py                     # Document ingestion
    ├── index_builder.py              # Chroma vector indexing
    ├── query_engine.py               # LLM-powered Q&A
    └── config.py                     # API & embedding config
```

---

## Quick Start

```bash
# Project 1 — Dashboard
cd project-1-dashboard && pip install -r requirements.txt && streamlit run app.py

# Project 2 — ML Pipeline
cd project-2-ml-pipeline && pip install -r requirements.txt && python train.py

# Project 3 — Recommender
cd project-3-recommender && pip install -r requirements.txt && python train.py

# Run all tests
pytest ds-portfolio/ -v
```
