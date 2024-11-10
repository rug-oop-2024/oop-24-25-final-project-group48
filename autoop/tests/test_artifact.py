import unittest
from unittest.mock import MagicMock
from autoop.core.ml.artifact import Artifact


class TestArtifact(unittest.TestCase):
    """
    This is a test for Artifact class.
    """

    def setUp(self) -> None:
        """
        Set up a mock object with MagicMock, and initialise the mocked class.
        """
        self.type = MagicMock
        self.data = MagicMock
        self.name = MagicMock
        self.asset_path = "path"

        self.artifact = Artifact(self.type, self.data, self.name,
                                 self.asset_path)

    def test_add_elements(self) -> None:
        """
        Compare the output of element addition of the Mock and class.
        """
        self.artifact.add_elements()
        self.assertEqual(self.artifact.dictionary.get("asset_path"),
                         self.asset_path)
        self.assertEqual(self.artifact.dictionary.get("version"),
                         "1.0.0")
        self.assertEqual(self.artifact.dictionary.get("data"),
                         self.data)
        self.assertEqual(self.artifact.dictionary.get("metadata"),
                         {})
        self.assertEqual(self.artifact.dictionary.get("tags"),
                         ['machine_learning'])
        self.assertEqual(self.artifact.dictionary.get("type"),
                         self.type)

    def test_read(self) -> None:
        """
        Compare the output of reading of the Mock and class.
        """
        data = self.artifact.read()
        self.assertEqual(data, self.data)

    def test_save(self) -> None:
        """
        Compare the output of saving of the Mock and class.
        """
        new_data = MagicMock
        self.artifact.save(new_data)
        self.assertEqual(self.artifact.data, new_data)
