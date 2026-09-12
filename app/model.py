from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

from app.preprocessing import preprocess_features


MODEL_NAME = "breast_cancer_svm"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "svm.joblib"


class SVMModel:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.pipeline: Pipeline | None = None

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model tại {self.model_path}")
        self.pipeline = joblib.load(self.model_path)

    def predict(self, features: list[float]) -> tuple[str, float, float]:
        if self.pipeline is None:
            raise RuntimeError("Model chưa được nạp")

        values = preprocess_features(features)
        raw_prediction = self.pipeline.predict(values)[0]
        prediction = str(raw_prediction)
        probabilities = self.pipeline.predict_proba(values)[0]
        classes = self.pipeline.classes_
        predicted_index = list(classes).index(raw_prediction)
        probability = float(probabilities[predicted_index])
        decision = float(np.asarray(self.pipeline.decision_function(values)).reshape(-1)[0])
        return prediction, probability, decision
