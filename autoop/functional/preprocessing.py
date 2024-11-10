from typing import List, Tuple
from autoop.core.ml.feature import Feature
from autoop.core.ml.dataset import Dataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_features(features: List[Feature], dataset: Dataset
                        ) -> List[Tuple[str, np.ndarray, dict]]:
    """
    Preprocess features.

    Args:
        features (List[Feature]): List of features.
        dataset (Dataset): Dataset object.

    Returns:
        List[str, Tuple[np.ndarray, dict]]: List of preprocessed features.
        Each ndarray of shape (N, ...)
    """
    results = []
    raw = dataset.read()

    for feature in features:
        if feature.type == "categorical":
            encoder = OneHotEncoder(sparse_output=False)
            # do not one-hot encode target
            if feature.name == "label":
                data = raw[feature.name].values
                artifact = {"type": "Label", "encoder": None}
            else:
                data = encoder.fit_transform(
                    raw[feature.name].values.reshape(-1, 1))
                # store the actual encoder object
                artifact = {"type": "OneHotEncoder", "encoder": encoder}
            results.append((feature.name, data, artifact))
        elif feature.type == "continuous":
            scaler = StandardScaler()
            data = scaler.fit_transform(
                raw[feature.name].values.reshape(-1, 1))
            artifact = {"type": "StandardScaler", "scaler": scaler}
            results.append((feature.name, data, artifact))

    return sorted(results, key=lambda x: x[0])
