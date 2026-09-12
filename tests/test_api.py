from fastapi.testclient import TestClient

from app.main import app


def test_health_and_prediction() -> None:
    with TestClient(app) as client:
        health_response = client.get("/health")
        prediction_response = client.post(
            "/api/v1/predict", json={"features": [0.2, -1.1, 0.5, 1.3]}
        )

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "healthy",
        "model_loaded": True,
        "model": "support_vector_machine",
    }
    assert prediction_response.status_code == 200
    body = prediction_response.json()
    assert body["success"] is True
    assert body["data"]["model"] == "support_vector_machine"
    assert body["data"]["prediction"] in {"0", "1"}
    assert 0 <= body["data"]["probability"] <= 1
    assert isinstance(body["data"]["decision_function"], float)


def test_invalid_feature_count_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/predict", json={"features": [1, 2, 3]})

    assert response.status_code == 422
