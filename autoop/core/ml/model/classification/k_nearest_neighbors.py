from autoop.core.ml.model import Model
from autoop.core.ml.artifact import Artifact
from sklearn import neighbors
import numpy as np


class KNN(Model):
    """
    This class uses a general ML model with two main methods,
    fit and predict. This model creates a wrapper around
    a linear model called KNN, where one attempts to predict
    the y-values for x-observations.
    """
    def __init__(self) -> None:
        """
        Inherits the constructor from its parent class,
        and initializes an attribute, which is a Ridge class.
        """
        super().__init__()
        self._classification = neighbors.KNeighborsClassifier()
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Implements the KNN fit method on the observations and ground
        truth, and saves the KNN parameters in the 'parameters' dictionary.

        Args:
            observations (np.ndarray): Data values on x-axis.
            ground_truth (np.ndarray): Corresponding y-values.
        """
        self._classification.fit(observations, ground_truth)
        n_neighbors = self._classification.get_params()["n_neighbors"]
        algorithm = self._classification.get_params()["algorithm"]
        self._parameters = {
            "n_neighbors": n_neighbors,
            "algorithm": algorithm
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Implements the KNN predict method.

        Args:
            observations (np.ndarray): Data values on x-axis.

        Returns:
            np.ndarray: The corresponding y-values.
        """
        predictions = self._classification.predict(observations)
        return np.array(predictions)

    def to_artifact(self, name: str) -> Artifact:
        """
        Converts the model into an Artifact.

        Args:
            name (str): Name of the Artifact.
        """
        return Artifact(type=self.type, data=self._parameters, name=name)
