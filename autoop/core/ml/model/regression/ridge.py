from autoop.core.ml.model import Model
from autoop.core.ml.artifact import Artifact
from sklearn import linear_model
import numpy as np


class Ridge(Model):
    """
    This class uses a general ML model with two main methods,
    fit and predict. This model creates a wrapper around
    a linear model called Ridge, where one attempts to predict
    the y-values for x-observations.
    """
    def __init__(self) -> None:
        """
        Initializes a Ridge model by inheriting the constructor from
        its parent class, and by creating model specific attributes.
        """
        super().__init__()
        self.regression_analysis = linear_model.Ridge()
        self._type = "regression"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Implements the Ridge fit method on the observations and ground
        truth, and saves the Ridge parameters in the 'parameters' dictionary.

        Args:
            observations (np.ndarray): Data values on x-axis.
            ground_truth (np.ndarray): Corresponding y-values.
        """
        self.regression_analysis.fit(observations, ground_truth)
        coef_ = self.regression_analysis.coef_
        intercept_ = self.regression_analysis.intercept_
        alpha = self.regression_analysis.get_params()["alpha"]
        solver = self.regression_analysis.get_params()["solver"]
        self._parameters = {
            "coef_": coef_,
            "intercept_": intercept_,
            "alpha": alpha,
            "solver": solver
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Implements the Ridge predict method.

        Args:
            observations (np.ndarray): X-observations of dataset.

        Returns:
            np.ndarray: Predictions.
        """
        predictions = self.regression_analysis.predict(observations)
        return np.array(predictions)

    def to_artifact(self, name: str) -> Artifact:
        """
        Converts the model into an Artifact.

        Args:
            name (str): Name of the Artifact.
        """
        return Artifact(type=self.type, data=self._parameters, name=name)
