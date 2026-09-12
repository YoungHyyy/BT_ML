from typing import Sequence

import numpy as np


EXPECTED_FEATURE_COUNT = 4


def preprocess_features(features: Sequence[float]) -> np.ndarray:
    """Validate and shape request features for the trained pipeline."""
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"features phải có đúng {EXPECTED_FEATURE_COUNT} giá trị")

    values = np.asarray(features, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("features phải là các số hữu hạn")
    return values.reshape(1, EXPECTED_FEATURE_COUNT)
