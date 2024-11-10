from sklearn.datasets import fetch_openml
import unittest
import pandas as pd

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.regression.elastic_net import ElasticNet
from autoop.core.ml.metric import MetricMSE


class TestPipeline(unittest.TestCase):
    """
    This is a test for Pipeline class.
    """

    def setUp(self) -> None:
        """
        Simulates the process of Pipeline by loading a dataset,
        saving it as a Dataset, checking for feature types, and initialising
        a Pipeline class.
        """
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=ElasticNet(),
            input_features=list(filter(lambda x: x.name != "age",
                                       self.features)),
            target_feature=Feature(name="age", type="continuous"),
            metrics=[MetricMSE()],
            split=0.8
        )
        self.ds_size = data.data.shape[0]

    def test_init(self) -> None:
        """
        Checks for instantiation of the correct object.
        """
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self) -> None:
        """
        Calls the preprocess_features function, and tests if the lengths
        of artifacts and features are the same.
        """
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self) -> None:
        """
        Calls the preprocess_features function, and split_data function,
        and checks for correct size and dimensions of the the training
        and testing sets.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.assertEqual(self.pipeline._train_X[0].shape[0],
                         int(0.8 * self.ds_size))
        self.assertEqual(self.pipeline._test_X[0].shape[0],
                         self.ds_size - int(0.8 * self.ds_size))

    def test_train(self) -> None:
        """
        Calls the preprocess_features function, and split_data function,
        and train, and checks that the parameters dictionary is not empty.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.parameters)

    def test_evaluate(self) -> None:
        """
        Calls the preprocess_features function, and split_data function,
        and train, and evaluate, and checks that predictions and metric
        results are not None, and that the results are of length 1.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.pipeline._evaluate()
        self.assertIsNotNone(self.pipeline._predictions)
        self.assertIsNotNone(self.pipeline._metrics_results)
        self.assertEqual(len(self.pipeline._metrics_results), 1)

    def test_evaluate_train(self) -> None:
        """
        Calls the preprocess_features function, and split_data function,
        and train, and evaluate, and checks that predictions and metric
        results are not None, and that the results are of length 1.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.pipeline._evaluate_train()
        self.assertIsNotNone(self.pipeline._train_metrics_results)
        self.assertEqual(len(self.pipeline._train_metrics_results), 1)
