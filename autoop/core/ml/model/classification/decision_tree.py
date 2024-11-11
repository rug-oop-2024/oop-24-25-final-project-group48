from autoop.core.ml.model import Model
from autoop.core.ml.artifact import Artifact
from sklearn import tree
import numpy as np
from typing import Optional


class DecisionTree(Model):
    """
    This class uses a general ML model with two main methods,
    fit and predict. This model creates a wrapper around
    a linear model called Decision Tree, where one attempts to predict
    the y-values for x-observations.
    """
    def __init__(self,
                 criterion: str = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1) -> None:
        """
        Inherits the constructor from its parent class,
        and initializes an attribute, which is a Ridge class.

        Args:
            criterion (str): The function to measure the quality of a split.
                             Supported criteria are "gini" for the Gini
                             impurity and "entropy" for the information gain.
            max_depth (Optional[int]): The maximum depth of the tree. If None,
                             nodes are expanded until all leaves are pure or
                             until all leaves contain less than
                             min_samples_split samples.
            min_samples_split (int): The minimum number of samples required
                            to split an internal node.
            min_samples_leaf (int): The minimum number of samples required
                            to be at a leaf node.
        """
        super().__init__()
        self._classification = tree.DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self._type = "classification"

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Implements the Decision Tree fit method on the observations and ground
        truth, and saves the SVC parameters in the 'parameters' dictionary.

        Args:
            observations (np.ndarray): Data values on x-axis.
            ground_truth (np.ndarray): Corresponding y-values.
        """
        self._classification.fit(observations, ground_truth)
        self._parameters = {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf
        }

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Implements the SVC predict method.

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
