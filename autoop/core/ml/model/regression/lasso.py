from autoop.core.ml.model import Model
from autoop.core.ml.artifact import Artifact
from sklearn import linear_model
import numpy as np


class Lasso(Model):
    """
    This class uses a general ML model with two main methods,
    fit and predict. This model creates a wrapper around
    a linear model called Lasso, where one attempts to predict
    the y-values for x-observations.
    """
    def __init__(self) -> None:
        """
        Inherits the constructor from its parent class,
        and initializes an attribute, which is a Lasso class.
        """
        super().__init__()
        self._regression_analysis = linear_model.Lasso()
        self._type = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Implements the Lasso fit method on the observations and ground
        truth, and saves the Lasso parameters in the 'parameters' dictionary.

        Args:
            observations (np.ndarray): Data values on x-axis.
            ground_truth (np.ndarray): Corresponding y-values.
        """
        self._regression_analysis.fit(observations, ground_truth)
        coef_ = self._regression_analysis.coef_
        intercept_ = self._regression_analysis.intercept_
        alpha = self._regression_analysis.get_params()["alpha"]
        self._parameters = {
            "coef_": coef_,
            "intercept_": intercept_,
            "alpha": alpha
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Implements the Lasso predict method.

        Args:
            observations (np.ndarray): Data values on x-axis.

        Returns:
            np.ndarray: The corresponding y-values.
        """
        predictions = self._regression_analysis.predict(observations)
        return np.array(predictions)

    def to_artifact(self, name: str) -> Artifact:
        """
        Converts the model into an Artifact.

        Args:
            name (str): Name of the Artifact.
        """
        return Artifact(type=self.type, data=self._parameters, name=name)
