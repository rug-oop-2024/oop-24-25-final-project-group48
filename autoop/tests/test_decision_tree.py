import unittest
from autoop.core.ml.model import DecisionTree
from autoop.core.ml.artifact import Artifact


class TestDecisionTree(unittest.TestCase):
    """
    This is a test for Decision Tree model class.
    """

    def setUp(self) -> None:
        """
        Initialises the tested class.
        """
        self.elastic_net = DecisionTree()

    def test_parameters(self) -> None:
        """
        Checks for instantiation of the correct object.
        """
        self.elastic_net._parameters = {"param": "value"}
        parameters = self.elastic_net.parameters
        self.assertIsInstance(parameters, dict)

    def test_fit(self) -> None:
        """
        This function is a wrapper around an existing library,
        it is expected to be functional.
        """
        pass

    def test_predict(self) -> None:
        """
        This function is a wrapper around an existing library,
        it is expected to be functional.
        """
        pass

    def test_to_artifact(self) -> None:
        """
        Checks for instantiation of the correct object.
        """
        self.elastic_net._parameters = {"param": "value"}
        name = "name"
        artifact = self.elastic_net.to_artifact(name)
        self.assertIsInstance(artifact, Artifact)
