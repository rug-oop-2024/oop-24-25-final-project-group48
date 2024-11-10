from typing import List, Dict, Any
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import OneHotEncoder


class Pipeline:
    """
    Represents a machine learning pipeline for preprocessing,
    training, and evaluation.
    """
    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split=0.8,
                 ) -> None:
        """
        Initializes the Pipeline.

        Args:
            metrics (List[Metric]): List of evaluation metrics.
            dataset (Dataset): The dataset to use.
            model (Model): The model to train and evaluate.
            input_features (List[Feature]): Features to be
            used as input.
            target_feature (Feature): The target feature to predict.
            split (float, optional): Train-test split ratio.
            Defaults to 0.8.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split

    def __str__(self) -> str:
        """
        Returns a string representation of the pipeline.

        Returns:
            str: String representation of the pipeline.
        """
        return f"""
                Pipeline(
                    model={self._model.type},
                    input_features={list(map(str, self._input_features))},
                    target_feature={str(self._target_feature)},
                    split={self._split},
                    metrics={list(map(str, self._metrics))},
                )
                """

    @property
    def model(self) -> Model:
        """
        Returns the model used in the pipeline.

        Returns:
            Model: The machine learning model.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during the pipeline
        execution to be saved

        Returns:
            List[Artifact]: List of artifacts for saving or inspection.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Dict[str, Any]) -> None:
        """
        Registers an artifact for the pipeline.

        Args:
            name (str): Name of the artifact.
            artifact (dict): The artifact to register.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses the input and target features, registering any
        artifacts generated.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset)[0]

        # check if target is one-hot encoded, convert back if so
        if artifact["type"] == "OneHotEncoder":
            target_data = np.argmax(target_data, axis=1)

        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)

        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)

        # get the input vectors and output vector
        self._output_vector = target_data
        self._input_vectors = [data for (
            feature_name, data, artifact) in input_results]

    def _split_data(self) -> None:
        """
        Splits the dataset into training and testing sets.
        """
        split = self._split
        self._train_X = [vector[:int(split * len(vector)
                                     )] for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)
                                   ):] for vector in self._input_vectors]
        self._train_y = self._output_vector[:int(split * len(
            self._output_vector))]
        self._test_y = self._output_vector[int(split * len(
            self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Combines multiple feature vectors into a single matrix.

        Args:
            vectors (List[np.ndarray]): List of feature vectors.

        Returns:
            np.ndarray: Combined matrix of features.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model using the training data.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y

        # check if the matrix has been one-hot encoded by checking
        # its dimensionality
        if Y.ndim == 2 and Y.shape[1] > 1:
            # retrieve the encoder for the target feature
            encoder = self._artifacts[self._target_feature.name]["encoder"]
            if isinstance(encoder, OneHotEncoder):
                # convert one-hot back to class labels
                Y = np.argmax(Y, axis=1)
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluates the model on the test set using the specified metrics.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._test_metrics_results = []
        predictions = self._model.predict(X)
        # decode predictions back to human-readable
        # labels if using OneHotEncoder
        encoder = self._artifacts[self._target_feature.name].get("encoder")
        # for regression, skip the decoding step
        if encoder and isinstance(encoder, OneHotEncoder):
            # convert class indices to one-hot encoded format
            num_classes = encoder.categories_[0].size
            # one-hot encode predictions
            predictions_one_hot = np.eye(num_classes)[predictions]
            Y_one_hot = np.eye(num_classes)[Y]
            predictions = encoder.inverse_transform(predictions_one_hot)
            Y = encoder.inverse_transform(Y_one_hot)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._test_metrics_results.append((metric, result))
        self._predictions = predictions

    def _evaluate_train(self) -> None:
        """
        Evaluates the model on the training set.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._train_metrics_results = []
        predictions = self._model.predict(X)
        # decode predictions back to human-readable
        # labels if using OneHotEncoder
        encoder = self._artifacts[self._target_feature.name].get("encoder")
        if encoder and isinstance(encoder, OneHotEncoder):
            # convert class indices to one-hot encoded format
            num_classes = encoder.categories_[0].size
            # one-hot encode predictions
            predictions_one_hot = np.eye(num_classes)[predictions]
            Y_one_hot = np.eye(num_classes)[Y]

            predictions = encoder.inverse_transform(predictions_one_hot)
            Y = encoder.inverse_transform(Y_one_hot)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._train_metrics_results.append((metric, result))

    def execute(self) -> Dict[str, List[Any]]:
        """
        Executes the pipeline, including preprocessing, training,
        and evaluation.

        Returns:
            Dict[List]: Dictionary containing test metrics,
            train metrics, and predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        self._evaluate_train()
        return deepcopy({
            "test metrics": self._test_metrics_results,
            "train metrics": self._train_metrics_results,
            "predictions": self._predictions,
        })
