from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.model import MODEL_NAME, SVMModel
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse


svm_model = SVMModel()


@asynccontextmanager
async def lifespan(_: FastAPI):
    svm_model.load()
    yield


app = FastAPI(
    title="Breast Tumor Classification Support API",
    version="1.0.0",
    description="API hỗ trợ chẩn đoán sơ bộ khối u lành tính hoặc ác tính bằng mô hình SVM. Kết quả chỉ là hỗ trợ chẩn đoán và không thay thế chẩn đoán lâm sàng của bác sĩ.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if svm_model.is_loaded else "unhealthy",
        model_loaded=svm_model.is_loaded,
        model=MODEL_NAME,
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        prediction, probability, decision = svm_model.predict(request.features)
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error

    if prediction == "0":
        diagnosis_message = f"Hỗ trợ chẩn đoán: khối u lành tính với độ tin cậy {probability:.2%}"
    else:
        diagnosis_message = f"Hỗ trợ chẩn đoán: khối u ác tính với độ tin cậy {probability:.2%}"

    return PredictionResponse(
        success=True,
        status=200,
        message=diagnosis_message,
        data={
            "model": MODEL_NAME,
            "endpoint": "/api/v1/predict",
            "prediction": prediction,
            "probability": probability,
            "decision_function": decision,
            "health_status": "healthy",
        },
    )
