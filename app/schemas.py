from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=4, max_length=4, description="Bốn đặc trưng dạng số"
    )


class PredictionData(BaseModel):
    model: str
    endpoint: str
    prediction: str
    probability: float
    decision_function: float
    health_status: str


class PredictionResponse(BaseModel):
    success: bool
    status: int
    message: str
    data: PredictionData


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model: str
