import unittest

from autoop.core.storage import LocalStorage, NotFoundError
import random
import tempfile


class TestStorage(unittest.TestCase):
    """
    This is a test for Storage class.
    """

    def setUp(self) -> None:
        """
        Initialises the tested class, and creates a temporary directory.
        """
        temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(temp_dir)

    def test_init(self) -> None:
        """
        Checks for instantiation of the correct object.
        """
        self.assertIsInstance(self.storage, LocalStorage)

    def test_store(self) -> None:
        """
        Tests the 'save' and 'load' functionality of the storage system, by
        saving a byte array to a specific key in the storage, verifying
        that the saved data can be loaded correctly using the same key.
        Checks if a NotFoundError is raised when loading from a non-existent
        key.

        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = "test/path"
        self.storage.save(test_bytes, key)
        self.assertEqual(self.storage.load(key), test_bytes)
        otherkey = "test/otherpath"
        # should not be the same
        try:
            self.storage.load(otherkey)
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_delete(self) -> None:
        """
        Tests the 'delete' method of the storage class, similarly like
        test_store does.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = "test/path"
        self.storage.save(test_bytes, key)
        self.storage.delete(key)
        try:
            self.assertIsNone(self.storage.load(key))
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_list(self) -> None:
        """
        Test the 'list' functionality of the storage system, by savinh byte
        arrays to multiple randomly generated keys under a common prefix.
        It lists all keys under the specified prefix, and verifies
        that the listed keys match the saved keys.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        random_keys = [f"test/{random.randint(0, 100)}" for _ in range(10)]
        for key in random_keys:
            self.storage.save(test_bytes, key)
        keys = self.storage.list("test")
        keys = ["/".join(key.split("/")[-2:]) for key in keys]
        self.assertEqual(set(keys), set(random_keys))
