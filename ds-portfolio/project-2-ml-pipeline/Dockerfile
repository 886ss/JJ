FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建模型目录
RUN mkdir -p models results data

# 预训练模型（构建时运行）
RUN python train.py --model logistic --C 1.0

# 暴露端口
EXPOSE 8000

# 启动 API 服务
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
