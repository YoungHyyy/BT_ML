from pathlib import Path

import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "svm.joblib"


def train_model() -> None:
    features, labels = make_classification(
        n_samples=600,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42,
    )
    train_features, _, train_labels, _ = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
        ]
    )
    pipeline.fit(train_features, train_labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
