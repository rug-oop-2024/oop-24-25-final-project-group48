from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from copy import deepcopy


def detect_feature_types(dataset: Dataset) -> deepcopy:
    """
    Detects feature types of a given dataset with the assumption of only
    categorical and numerical features and no NaN values.

    Args:
        dataset: Dataset

    Returns:
        deepcopy: Deepcopy of a list of features with their types.
    """

    features_list = list()
    df_dataset = dataset.read()
    continuous_data = df_dataset.select_dtypes(include=["float", "int"])
    # categorical data can be True/False, or other non-int, and non-float
    # type of objects, leading to 'bool', and 'object' dtypes
    categorical_data = df_dataset.select_dtypes(include=["object", "bool"])

    for feature_name in list(continuous_data.columns):
        feature_type = "continuous"
        feature = Feature(feature_name, feature_type)
        features_list.append(feature)

    for feature_name in list(categorical_data.columns):
        feature_type = "categorical"
        feature = Feature(feature_name, feature_type)
        features_list.append(feature)

    return deepcopy(features_list)
