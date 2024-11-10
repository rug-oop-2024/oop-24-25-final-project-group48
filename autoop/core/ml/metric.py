from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union
from copy import deepcopy


# lists to save the metrics, and use them to print out options in Streamlit
CLASSIFICATION_METRICS = [
    "Accuracy",
    'Specificity',
    "Hamming Loss"
]

REGRESSION_METRICS = [
    "Mean Squared Error",
    'Mean Absolute Error',
    'Root Mean Squared Error'
]


class Metric(ABC):
    """
    An abstract base class for a Metric.
    """
    @abstractmethod
    def __str__(self) -> str:
        """
        Prompts the user to override the __str__ magic method
        in order to print the name of the Metric.

        Returns:
            str: Name of the metric.
        """
        pass

    @abstractmethod
    def evaluate(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates a metric.

        Args:
            predictions (np.ndarray): Predictions given by a model.
            ground_truth (np.ndarray): The ground truth of a dataset.

        Returns:
            float: The metric result.
        """
        pass


class MetricMSE(Metric):
    """
    A Metric class for calculating Mean Squared Error.
    """
    def __str__(self) -> str:
        """
        Prints the name of the Metric.

        Returns:
            str: Name of the Metric.
        """
        return 'Mean Squared Error'

    def evaluate(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates a metric.

        Args:
            predictions (np.ndarray): Predictions given by a model.
            ground_truth (np.ndarray): The ground truth of a dataset.

        Returns:
            float: The metric result.
        """
        squared_error = (ground_truth - predictions) ** 2
        mse = np.mean(squared_error)
        return mse


class MetricMAE(Metric):
    """
    A Metric class for calculating Mean Average Error.
    """
    def __str__(self) -> str:
        """
        Prints the name of the Metric.

        Returns:
            str: Name of the Metric.
        """
        return 'Mean Absolute Error'

    def evaluate(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates a metric.

        Args:
            predictions (np.ndarray): Predictions given by a model.
            ground_truth (np.ndarray): The ground truth of a dataset.

        Returns:
            float: The metric result.
        """
        abs_error = np.abs(ground_truth - predictions)
        mae = np.mean(abs_error)
        return mae


class MetricRMSE(Metric):
    """
    A Metric class for calculating Root Mean Squared Error.
    """
    def __str__(self) -> str:
        """
        Prints the name of the Metric.

        Returns:
            str: Name of the Metric.
        """
        return 'Root Mean Squared Error'

    def evaluate(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates a metric.

        Args:
            predictions (np.ndarray): Predictions given by a model.
            ground_truth (np.ndarray): The ground truth of a dataset.

        Returns:
            float: The metric result.
        """
        squared_error = (ground_truth - predictions) ** 2
        mse = np.mean(squared_error)
        rmse = np.sqrt(mse)
        return rmse


class ConfusionMatrix:
    """
    A class for creating a Confusion Matrix.
    """

    def __init__(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> None:
        """
        Initializes a Confusion Matrix using predictions created by a model,
        the ground truth of the dataset, and the positive and negative
        by finding the unique elements of an array.

        Args:
            predictions (np.ndarray): Predictions given by a model.
            ground_truth (np.ndarray): The ground truth of a dataset.
        """
        self.predictions = predictions
        self.ground_truth = ground_truth

        # Infer positive and negative from unique labels
        unique_labels = np.unique(np.concatenate([predictions, ground_truth]))
        self.negative, self.positive = unique_labels[0], unique_labels[1]

    def true_positive(self) -> Union[int, float]:
        """
        Returns a true positive in a Confusion Matrix.

        Returns:
            np.ndarray: The sum of arrays.
        """
        return np.sum(np.logical_and(self.predictions == self.positive,
                                     self.ground_truth == self.positive))

    def true_negative(self) -> Union[int, float]:
        """
        Returns a true negative in a Confusion Matrix.

        Returns:
            np.ndarray: The sum of arrays.
        """
        return np.sum(np.logical_and(self.predictions == self.negative,
                                     self.ground_truth == self.negative))

    def false_positive(self) -> Union[int, float]:
        """
        Returns a false positive in a Confusion Matrix.

        Returns:
            np.ndarray: The sum of arrays.
        """
        return np.sum(np.logical_and(self.predictions == self.positive,
                                     self.ground_truth == self.negative))

    def false_negative(self) -> Union[int, float]:
        """
        Returns a false negative in a Confusion Matrix.

        Returns:
            np.ndarray: The sum of arrays.
        """
        return np.sum(np.logical_and(self.predictions == self.negative,
                                     self.ground_truth == self.positive))


class MetricAccuracy(Metric):
    """
    A Metric class for calculating Accuracy.
    """
    def __str__(self) -> str:
        """
        Prints the name of the Metric.

        Returns:
            str: Name of the Metric.
        """
        return 'Accuracy'

    def evaluate(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates a metric.

        Args:
            predictions (np.ndarray): Predictions given by a model.
            ground_truth (np.ndarray): The ground truth of a dataset.

        Returns:
            float: The metric result.
        """
        cm = ConfusionMatrix(predictions, ground_truth)
        if len(predictions) > 0:
            Accuracy = (cm.true_negative() + cm.true_positive()
                        ) / len(predictions)
        else:
            Accuracy = 0.0
        return Accuracy


class MetricSpecificity(Metric):
    """
    A Metric class for calculating Specificity.
    """
    def __str__(self) -> str:
        """
        Prints the name of the Metric.

        Returns:
            str: Name of the Metric.
        """
        return 'Specificity'

    def evaluate(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates a metric.

        Args:
            predictions (np.ndarray): Predictions given by a model.
            ground_truth (np.ndarray): The ground truth of a dataset.

        Returns:
            float: The metric result.
        """
        cm = ConfusionMatrix(predictions, ground_truth)
        denominator = cm.true_negative() + cm.false_positive()
        if len(predictions) != 0:
            if denominator == 0:
                raise ZeroDivisionError('Cannot divide by zero.')
            Specificity = cm.true_negative() / denominator
        else:
            return 0.0
        return Specificity


class MetricHammingLoss(Metric):
    """
    A Metric class for calculating Hamming Loss.
    """
    def __str__(self) -> str:
        return 'Hamming Loss'

    def evaluate(self, predictions: np.ndarray,
                 ground_truth: np.ndarray) -> float:
        """
        Calculates a metric.

        Args:
            predictions (np.ndarray): Predictions given by the model.
            ground_truth (np.ndarray): The ground truth of a dataset.

        Returns:
            float: The metric result.
        """
        cm = ConfusionMatrix(predictions, ground_truth)
        total_instances = len(predictions)
        if len(predictions) != 0:
            hamming_loss = (cm.false_positive() + cm.false_negative()
                            ) / total_instances
        else:
            raise ZeroDivisionError('Cannot divide by zero.')
        return hamming_loss


def get_metric(name: List[str]) -> List[Metric]:
    """
    Factory function to get a metric by name. Returns a list of metrics,
    since more of them can be chosen, and we should able to loop through
    them in a Pipeline instance.

    Args:
        name (str): Name of the metric.

    Raises:
        ValueError: When a given name is not a metric option.

    Returns:
        List: List of the chosen metrics.
    """
    list_of_metrics = []
    for el in name:
        if el == 'Mean Squared Error':
            list_of_metrics.append(MetricMSE())
        elif el == 'Mean Absolute Error':
            list_of_metrics.append(MetricMAE())
        elif el == 'Root Mean Squared Error':
            list_of_metrics.append(MetricRMSE())
        elif el == 'Accuracy':
            list_of_metrics.append(MetricAccuracy())
        elif el == 'Specificity':
            list_of_metrics.append(MetricSpecificity())
        elif el == "Hamming Loss":
            list_of_metrics.append(MetricHammingLoss())
        else:
            raise ValueError(f"Metric {el} is not an option.")
    return deepcopy(list_of_metrics)
