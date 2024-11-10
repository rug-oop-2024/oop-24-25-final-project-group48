from autoop.core.ml.artifact import Artifact
from typing import Any
import pandas as pd
import io


class Dataset(Artifact):
    """
    Represents a dataset artifact, allowing for storage
    and retrieval of datasets.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes a Dataset artifact.

        Args:
            *args (Any): Positional arguments passed to the
            Artifact constructor.
            **kwargs (Any): Keyword arguments passed to the
            Artifact constructor.
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str,
                       version: str = "1.0.0") -> 'Dataset':
        """
        Creates a Dataset artifact from a DataFrame.

        Args:
            data (pd.DataFrame): The dataset as a DataFrame.
            name (str): The name of the dataset.
            asset_path (str): The file path where the dataset
            is stored.
            version (str): The version of the dataset. Defaults
            to "1.0.0".

        Returns:
            Dataset: An instance of the Dataset class containing
            the serialized data.
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        Reads the dataset from its stored format and returns
        it as a DataFrame.

        Returns:
            pd.DataFrame: The dataset loaded as a DataFrame.
        """
        bytes_data = super().read()
        csv = bytes_data.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Saves a DataFrame as a dataset artifact in serialized format.

        Args:
            data (pd.DataFrame): The dataset to be saved.

        Returns:
            bytes: The serialized form of the DataFrame.
        """
        bytes_data = data.to_csv(index=False).encode()
        return super().save(bytes_data)
