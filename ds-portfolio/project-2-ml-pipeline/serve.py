"""
FastAPI 推理服务

启动方式：
    uvicorn serve:app --host 0.0.0.0 --port 8000 --reload

测试：
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"text": "The new graphics card from NVIDIA offers incredible performance for gaming and AI workloads."}'

    http://localhost:8000/docs   — Swagger UI
"""

import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
from predict import load_artifacts, predict_single

# ---------- FastAPI 应用 ----------
app = FastAPI(
    title="新闻分类 API",
    description="基于 sklearn + TF-IDF 的新闻文本多分类服务",
    version="1.0.0",
)

# ---------- 全局模型加载 ----------
model = None
extractor = None


@app.on_event("startup")
def startup():
    global model, extractor
    print("[启动] 正在加载模型...")
    model, extractor = load_artifacts()
    print(f"[启动] OK 模型已就绪 | 类别: {list(model.classes_)}")


# ---------- 请求/响应模型 ----------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=10000,
                      description="待分类的英文新闻文本",
                      example="The new graphics card from NVIDIA offers incredible performance.")

class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_items=1, max_items=50,
                             description="待分类的文本列表")

class PredictResponse(BaseModel):
    prediction: str = Field(..., description="预测类别")
    probabilities: dict[str, float] | None = Field(None, description="各类别概率")

class BatchPredictResponse(BaseModel):
    predictions: list[str] = Field(..., description="预测类别列表")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str] | None


# ---------- 路由 ----------
@app.get("/", response_model=HealthResponse)
def root():
    """健康检查 + 基本信息"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "classes": list(model.classes_) if model else None,
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """健康检查"""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "classes": list(model.classes_) if model else None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """单条文本分类"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        result = predict_single(request.text, model, extractor)
        return PredictResponse(
            prediction=result["prediction"],
            probabilities=result["probabilities"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest):
    """批量文本分类"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        preds = []
        for text in request.texts:
            result = predict_single(text, model, extractor)
            preds.append(result["prediction"])
        return BatchPredictResponse(predictions=preds)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": traceback.format_exc()},
    )
