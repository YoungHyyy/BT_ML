from sklearn.datasets import load_breast_cancer
from fastapi.testclient import TestClient

from app.main import app


def test_health_and_prediction() -> None:
    dataset = load_breast_cancer()
    sample = dataset.data[0, :4].tolist()

    with TestClient(app) as client:
        health_response = client.get("/health")
        prediction_response = client.post("/api/v1/predict", json={"features": sample})

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "healthy",
        "model_loaded": True,
        "model": "breast_cancer_svm",
    }
    assert prediction_response.status_code == 200
    body = prediction_response.json()
    assert body["success"] is True
    assert body["data"]["model"] == "breast_cancer_svm"
    assert body["data"]["prediction"] in {"0", "1"}
    assert 0 <= body["data"]["probability"] <= 1
    assert isinstance(body["data"]["decision_function"], float)
    assert "Hỗ trợ chẩn đoán: khối u" in body["message"]
    assert "độ tin cậy" in body["message"]


def test_invalid_feature_count_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"features": [1, 2, 3]})

    assert response.status_code == 422
