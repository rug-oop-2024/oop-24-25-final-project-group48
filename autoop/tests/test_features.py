import unittest
from sklearn.datasets import load_iris, fetch_openml
import pandas as pd
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types


class TestFeatures(unittest.TestCase):
    """
    This is a test for Features class.
    """

    def setUp(self) -> None:
        """
        Set up a mock object with MagicMock, and initialise the mocked class
        in this case empty.
        """
        pass

    def test_detect_features_continuous(self) -> None:
        """
        Loads a dataset, and uses the DataFrame class to check for correct
        feature instance, length, name, and type.
        """
        iris = load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=df,
        )
        self.X = iris.data
        self.y = iris.target
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in iris.feature_names, True)
            self.assertEqual(feature.type, "continuous")

    def test_detect_features_with_categories(self) -> None:
        """
        Loads a dataset, and uses the DataFrame class to check for correct
        feature instance, length, name, and type.
        """
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        features = detect_feature_types(dataset)
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 14)
        continuous_columns = [
            "age",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        categorical_columns = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        for feature in features:
            self.assertIsInstance(feature, Feature)
            self.assertEqual(feature.name in data.feature_names, True)
        for detected_feature in filter(lambda x: x.name in continuous_columns,
                                       features):
            self.assertEqual(detected_feature.type, "continuous")
        for detected_feature in filter(lambda x: x.name in categorical_columns,
                                       features):
            self.assertEqual(detected_feature.type, "categorical")


if __name__ == '__main__':
    unittest.main()
