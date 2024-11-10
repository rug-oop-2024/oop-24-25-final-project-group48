from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from autoop.core.ml.artifact import Artifact


class Model(ABC):
    """
    This is a general ML model intended to provide a base
    structure for different models.
    """
    def __init__(self) -> None:
        """
        Initializes a Model with parameters, and type, which is based
        on the task type (classification/regression).
        """
        self._parameters = dict()
        self._type = ""

    @property
    def type(self) -> str:
        """
        Returns the type of Model.

        Returns:
            str: Type of the model.
        """
        return self._type

    @property
    def parameters(self) -> deepcopy:
        """
        Creates a deepcopy of the parameters.

        Returns:
            deepcopy: A copy of the parameters.
        """
        return deepcopy(self._parameters)

    @abstractmethod
    def fit(self, observations: np.ndarray,
            ground_truth: np.ndarray) -> None:
        """
        This function is intended to be modified for the purpose
        of fitting.

        Args:
            observations (np.ndarray): Data values on x-axis.
            ground_truth (np.ndarray): Corresponding y-values.
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        This function is intended to be modified for the purpose
        of predicting.

        Args:
            observations (np.ndarray): Data values on x-axis.

        Returns:
            np.ndarray: Predictions.
        """
        pass

    @abstractmethod
    def to_artifact(self, name: str) -> Artifact:
        """
        Converts the model into an Artifact.

        Args:
            name (str): Name of the Artifact.

        Returns:
            Artifact: An Artifact instance containing the model's
            parameter, and type.
        """
        pass
