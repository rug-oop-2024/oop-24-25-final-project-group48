from abc import ABC, abstractmethod
import os
from glob import glob
from copy import deepcopy


class NotFoundError(Exception):
    """
    Exception raised when a specified path is not found in storage.
    """
    def __init__(self, path: str) -> None:
        """
        Initializes the NotFoundError with the provided path.

        Args:
            path (str): The path that was not found.

        Returns:
            None
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Class for storage.
    """
    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Saves data to a given path.

        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Loads data from a given path.

        Args:
            path (str): Path to load data.

        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Deletes data at a given path.

        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        Lists all paths under a given path.

        Args:
            path (str): Path to list.

        Returns:
            list: List of paths.
        """
        pass


class LocalStorage(Storage):
    """
    A class for a storage system for saving, loading, deleting,
    and listing files locally.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes the LocalStorage with a base directory.

        Args:
            base_path (str): The base directory where files
            will be stored. Defaults to "./assets".
        """
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Saves data to a specified key.

        Args:
            data (bytes): The data to save.
            key (str): The relative path (key) where the data
            will be stored.
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Loads data from a specified key.

        Args:
            key (str): The relative path (key) from which to load the data.

        Returns:
            bytes: The data loaded from the specified key.

        Raises:
            NotFoundError: If the specified key does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Deletes the file at the specified key.

        Args:
            key (str): The relative path (key) to delete. Defaults to "/".

        Raises:
            NotFoundError: If the specified key does not exist.
        """
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str) -> deepcopy:
        """
        Lists all files under the specified prefix.

        Args:
            prefix (str): The prefix (directory) to list files from.

        Returns:
            deepcopy: A deepcopy of a list of file paths under
            the specified prefix.

        Raises:
            NotFoundError: If the specified prefix does not exist.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        return deepcopy(list(filter(os.path.isfile, keys)))

    def _assert_path_exists(self, path: str) -> None:
        """
        Asserts that the specified path exists.

        Args:
            path (str): The path to check.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Joins the base path with the given relative path.

        Args:
            path (str): The relative path to join with the base path.

        Returns:
            str: The full path.
        """
        return os.path.join(self._base_path, path)
