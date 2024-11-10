import unittest
import numpy as np
from autoop.core.ml.metric import MetricMSE, MetricMAE, MetricRMSE, \
    MetricAccuracy, MetricSpecificity, MetricHammingLoss


class TestMSE(unittest.TestCase):
    """
    This is a test for MSE metric class.
    """

    def setUp(self) -> None:
        """
        Initialises the tested class.
        """
        self.mse = MetricMSE()

    def test_evaluate(self) -> None:
        """
        Tests for correct calculation given sample predictions and ground
        truth.
        """
        predictions = np.array([1, 2])
        ground_truth = np.array([3, 4])
        test_mse = 4
        mse = self.mse.evaluate(predictions, ground_truth)
        self.assertEqual(mse, test_mse)


class TestMAE(unittest.TestCase):
    """
    This is a test for MAE metric class.
    """

    def setUp(self) -> None:
        """
        Initialises the tested class.
        """
        self.mae = MetricMAE()

    def test_evaluate(self) -> None:
        """
        Tests for correct calculation given sample predictions and ground
        truth.
        """
        predictions = np.array([1, 2])
        ground_truth = np.array([3, 4])
        test_mae = 2
        mae = self.mae.evaluate(predictions, ground_truth)
        self.assertEqual(mae, test_mae)


class TestRMSE(unittest.TestCase):
    """
    This is a test for RMSE metric class.
    """

    def setUp(self) -> None:
        """
        Initialises the tested class.
        """
        self.rmse = MetricRMSE()

    def test_evaluate(self) -> None:
        """
        Tests for correct calculation given sample predictions and ground
        truth.
        """
        predictions = np.array([1, 2])
        ground_truth = np.array([3, 4])
        test_rmse = 2
        rmse = self.rmse.evaluate(predictions, ground_truth)
        self.assertEqual(rmse, test_rmse)


class TestAccuracy(unittest.TestCase):
    """
    This is a test for Accuracy metric class.
    """

    def setUp(self) -> None:
        """
        Initialises the tested class.
        """
        self.accuracy = MetricAccuracy()

    def test_evaluate(self) -> None:
        """
        Tests for correct calculation given sample predictions and ground
        truth.
        """
        predictions = np.array([1, 2])
        ground_truth = np.array([3, 4])
        test_accuracy = 0
        accuracy = self.accuracy.evaluate(predictions, ground_truth)
        self.assertEqual(accuracy, test_accuracy)


class TestSpecificity(unittest.TestCase):
    """
    This is a test for Specificity metric class.
    """

    def setUp(self) -> None:
        """
        Initialises the tested class.
        """
        self.specificity = MetricSpecificity()

    def test_evaluate(self) -> None:
        """
        Tests for correct calculation given sample predictions and ground
        truth.
        """
        predictions = np.array([3, 4])
        ground_truth = np.array([3, 4])
        test_specificity = 1
        specificity = self.specificity.evaluate(predictions, ground_truth)
        self.assertEqual(specificity, test_specificity)


class TestHammingLoss(unittest.TestCase):
    """
    This is a test for hamming_loss metric class.
    """

    def setUp(self) -> None:
        """
        Initialises the tested class.
        """
        self.hamming_loss = MetricHammingLoss()

    def test_evaluate(self) -> None:
        """
        Tests for correct calculation given sample predictions and ground
        truth.
        """
        predictions = np.array([1, 2, 1, 2])
        ground_truth = np.array([1, 5, 3, 2])
        test_hamming_loss = 0
        hamming_loss = self.hamming_loss.evaluate(predictions, ground_truth)
        self.assertEqual(hamming_loss, test_hamming_loss)
