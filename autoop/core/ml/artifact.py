import base64
from typing import Optional
import pandas as pd


class Artifact:
    """
    A class for Artifact, and abstract object to refer to an asset
    and its information.
    """
    def __init__(self, type: str = None, data=None, name: str = None,
                 asset_path: str = None, version: str = "1.0.0",
                 tags: list = ['machine_learning'], metadata: dict = {}
                 ) -> None:
        """
        Initializes an Artifact, by creating a dictionary, with all
        essential data about an asset we are saving.

        Args:
            type (str): Type of asset. Defaults to None.
            data (Optional[pd.DataFrame]): The data of the asset.
            Defaults to None.
            name (str): Name of asset. Defaults to None.
            asset_path (str): Path to asset. Defaults to None.
            version (str): Version. Defaults to "1.0.0".
            tags (list): Semantic tags. Defaults to ['machine_learning'].
            metadata (dict): Metadata. Defaults to {}.
        """
        self.dictionary = {}
        self.type = type
        self.asset_path = asset_path
        self.version = version
        self.data = data
        self.name = name
        if self.asset_path is not None:
            self.id = f"{base64.b64encode(self.asset_path.encode()).decode()}"
        f":{self.version}"
        self.metadata = metadata
        self.tags = tags

    def add_elements(self) -> None:
        """
        Adds all initialized attributes to the internal dictionary.
        """
        self.dictionary['asset_path'] = self.asset_path
        self.dictionary['version'] = self.version
        self.dictionary['data'] = self.data
        self.dictionary['metadata'] = self.metadata
        self.dictionary['tags'] = self.tags
        self.dictionary['type'] = self.type

    def read(self) -> Optional[pd.DataFrame]:
        """
        Returns data of the asset.

        Returns:
            Optional[pd.DataFrame]: The data of the asset.
        """
        return self.data

    def save(self, data: bytes) -> None:
        """
        Saves data of the asset.

        Args:
            data (bytes): The data of the asset in bytes.
        """
        self.data = data
