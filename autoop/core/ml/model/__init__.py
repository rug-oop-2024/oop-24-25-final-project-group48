from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.elastic_net import ElasticNet
from autoop.core.ml.model.regression.lasso import Lasso
from autoop.core.ml.model.regression.ridge import Ridge
from autoop.core.ml.model.classification.k_nearest_neighbors import KNN
from autoop.core.ml.model.classification.stochastic_gradient_descent import SGD
from autoop.core.ml.model.classification.decision_tree import DecisionTree

# lists with models, to later use as options in streamlit
REGRESSION_MODELS = [
    "Elastic Net",
    "Lasso",
    "Ridge"
]

CLASSIFICATION_MODELS = [
    "K-Nearest Neighbors",
    "Stochastic Gradient Descent",
    "Decision Tree"
]


def get_model(model_name: str) -> Model:
    """
    Factory function to get a model by name.

    Args:
        model_name (str): Name of the model.

    Returns:
        Model: Instance of chosen Model.
    """
    if model_name == 'Elastic Net':
        return ElasticNet()
    elif model_name == 'Lasso':
        return Lasso()
    elif model_name == 'Ridge':
        return Ridge()
    elif model_name == 'K-Nearest Neighbors':
        return KNN()
    elif model_name == 'Stochastic Gradient Descent':
        return SGD()
    elif model_name == 'Decision Tree':
        return DecisionTree()
