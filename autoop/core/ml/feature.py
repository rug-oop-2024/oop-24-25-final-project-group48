from typing import Optional, Literal


class Feature:
    """
    A class for a Feature.
    """
    def __init__(self, feature_name: str, feature_type: str) -> None:
        """
        Initializes a Feature with a name and type.

        Args:
            feature_name (str): A name of the Feature.
            feature_type (str): Either a categorical or numerical.
        """
        self._name = feature_name
        self._type = feature_type

    @property
    def name(self) -> str:
        """
        Returns the attribute name.

        Returns:
            str: The name of the Feature.
        """
        return self._name

    @property
    def type(self) -> str:
        """
        Returns the attribute type.

        Returns:
            str: The type of the Feature.
        """
        return self._type

    @name.setter
    def name(self, new_name: str) -> Optional[ValueError]:
        """
        Sets the attribute name.

        Args:
            new_name (str): New name.

        Raises:
            ValueError: If the name given is not a string.

        Returns:
            Optional[ValueError]: Can return None, or an Error.
        """
        if type(new_name) is str:
            self._name = new_name
        else:
            raise ValueError('Name must be a string.')

    @type.setter
    def type(self, new_type: Literal['continuous', 'categorical']
             ) -> Optional[ValueError]:
        """
        Sets the attribute type.

        Args:
            new_type (str): New type.

        Raises:
            ValueError: If the type is not of the given type hint.

        Returns:
            Optional[ValueError]: Can return None, or an Error.
        """
        if new_type in ['continuous', 'categorical']:
            self._type = new_type
        else:
            raise ValueError("Type must be either 'categorical' or"
                             "'continuous")
