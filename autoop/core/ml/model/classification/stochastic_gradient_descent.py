from autoop.core.ml.model import Model
from autoop.core.ml.artifact import Artifact
from sklearn import linear_model
import numpy as np


class SGD(Model):
    """
    This class uses a general ML model with two main methods,
    fit and predict. This model creates a wrapper around
    a linear model called Stochastic Gradient Descent, where
    one attempts to predict the y-values for x-observations.
    """
    def __init__(self) -> None:
        """
        Inherits the constructor from its parent class,
        and initializes an attribute, which is a Ridge class.
        """
        super().__init__()
        self._classification = linear_model.SGDClassifier()
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Implements the Stochastic Gradient Descent fit method on the
        observations and ground truth, and saves the Stochastic
        Gradient Descent parameters in the 'parameters' dictionary.

        Args:
            observations (np.ndarray): Data values on x-axis.
            ground_truth (np.ndarray): Corresponding y-values.
        """
        self._classification.fit(observations, ground_truth)
        coef_ = self._classification.coef_
        intercept_ = self._classification.intercept_
        loss = self._classification.get_params()["loss"]
        self._parameters = {
            "coef_": coef_,
            "intercept_": intercept_,
            "loss": loss
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Implements the Stochastic Gradient Descent predict method.

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
