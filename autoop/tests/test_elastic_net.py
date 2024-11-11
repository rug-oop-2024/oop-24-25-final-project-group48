import unittest
from autoop.core.ml.model import ElasticNet
from autoop.core.ml.artifact import Artifact


class TestElasticNet(unittest.TestCase):
    """
    This is a test for Elastic Net model class.
    """

    def setUp(self) -> None:
        """
        Set up a mock object with MagicMock, and initialise the mocked class.
        """
        self.elastic_net = ElasticNet()

    def test_parameters(self) -> None:
        """
        Set up a mock object with MagicMock, and initialise the mocked class.
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
        self.elastic_net._parameters = {"param": "value"}
        name = "name"
        artifact = self.elastic_net.to_artifact(name)
        self.assertIsInstance(artifact, Artifact)
