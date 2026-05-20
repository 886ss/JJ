# 📊 电商数据分析 Dashboard

基于 Python + Streamlit 构建的交互式电商数据分析平台。

## 功能模块

| 模块 | 功能 |
|------|------|
| 📈 销售趋势 | 日销售额 / 订单量双轴图、月度环比、星期×时段热力图 |
| 🏷️ 商品分析 | 品类销售额排名、客单价对比、品类评分分布 |
| 🧑‍🤝‍🧑 客户画像 | RFM 分层、支付方式分布、地理分布、评分分布 |
| 📊 综合分析 | 交叉透视表、原始数据探查 |

## 数据说明

- 数据集：模拟生成的 12,000 条巴西电商交易数据
- 包含字段：订单 ID、客户 ID、商品品类、价格、数量、日期、状态、支付方式、州

## 技术栈

```
Python 3.10+
├── Streamlit  : Web 应用框架
├── Pandas     : 数据处理
├── NumPy      : 数值计算
├── Plotly     : 交互式可视化
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Dashboard
streamlit run app.py
```

## 项目结构

```
project-1-dashboard/
├── app.py              # Streamlit 主应用
├── data_loader.py      # 数据生成 / 加载 / 预处理
├── visualizations.py   # Plotly 图表函数
├── requirements.txt    # 依赖
└── README.md
```

## 简历亮点

> 独立开发电商数据分析 Dashboard，清洗并关联 12,000+ 条订单/用户/商品数据，构建 15 个分析维度；开发交互式数据面板，集成销售趋势、品类排名、地理分布等 8 个可视化组件，支持多条件动态筛选与 RFM 客户分层。
