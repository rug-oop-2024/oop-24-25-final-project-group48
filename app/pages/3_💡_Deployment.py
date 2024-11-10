import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.feature import Feature
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import \
    get_model, CLASSIFICATION_MODELS, REGRESSION_MODELS
from autoop.core.ml.metric import \
    get_metric, CLASSIFICATION_METRICS, REGRESSION_METRICS
from autoop.functional.feature import detect_feature_types
from copy import deepcopy


st.set_page_config(page_title="Deployment", page_icon="💡")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


# create a page introduction
st.write("# Deployment")
write_helper_text("In this section, you can view saved pipelines.")

st.subheader("Saved Pipelines")

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list(type="pipeline")

# allow the user to select existing pipelines
if not pipelines:
    st.warning("No saved pipelines found.")
else:
    pipeline_name = st.selectbox("Select a saved pipeline:",
                                 (artifact.name for artifact in pipelines))
# based on the selection show a pipeline summary
    selected_pipeline = next(
        (artifact for artifact in pipelines if artifact.name == pipeline_name),
        None
    )

    if selected_pipeline:
        st.subheader("Pipeline Summary")
        st.write(selected_pipeline.data.decode())

st.write("Predict")
uploaded_file = st.file_uploader(label="Upload your dataset", type='csv')
file_name = st.text_input("Name the Dataset:")

if uploaded_file and selected_pipeline:
    dataset = Dataset.from_dataframe(
        data=pd.read_csv(uploaded_file),
        name=file_name,
        asset_path="./assets/dbo"
    )

# check for correct instantiation
if isinstance(dataset, Dataset):
    features = detect_feature_types(dataset)
    input_features = []

    # create a table of all feature options
    data = {
        'Name': [feature.name for feature in features],
        'Type': [feature.type for feature in features]
    }

    # even though Name and Type of features are immutable, we
    # return a deepcopy as the dictionary could be modified
    df = pd.DataFrame(deepcopy(data))
    st.table(df)

    # select input features
    input_features_names = st.multiselect("Choose input features:",
                                          (feature.name for feature in features
                                           ))

    # select target feature
    target_feature_name = st.selectbox("Choose a target feature:",
                                       (feature.name for feature in features))

    if target_feature_name in input_features_names:
        st.write("Target feature cannot be input feature.")
    else:
        for feature in features:
            if feature.name in input_features_names:
                input_features.append(feature)
            elif feature.name == target_feature_name:
                target_feature = feature

        # check for correct instantiation
        if isinstance(target_feature, Feature):
            # we make sure to correctly choose the task type
            # for the model later on
            if target_feature.type == "categorical":
                task_type = "classification"
            elif target_feature.type == "continuous":
                task_type = "regression"

            # inform the user about the task type
            st.write("Your task type is:", task_type)

            st.subheader("Models")

            # prompt the user with model choices, according to
            # the task type
            if task_type == "regression":
                model_choice = str(st.selectbox("Select a model:",
                                                deepcopy(REGRESSION_MODELS)))
            elif task_type == "classification":
                model_choice = str(st.selectbox("Select a model:",
                                                deepcopy(CLASSIFICATION_MODELS)
                                                ))

            st.subheader("Pipeline")

            # prompt the user to choose a train-test split
            split = st.slider("Choose a train-test split value:", 0.1, 0.9,
                              step=0.1)

            # prompt the user with metric choices, according to
            # the task type
            if task_type == "regression":
                metric_choice = st.multiselect("Select a metric:",
                                               REGRESSION_METRICS)
            elif task_type == "classification":
                metric_choice = st.multiselect("Select a metric:",
                                               CLASSIFICATION_METRICS)

            chosen_model = get_model(model_choice)
            chosen_metric = get_metric(metric_choice)

            # print a formatted summary of the Pipeline for the user
            inp_feat = ", ".join(feature for feature in input_features_names)
            metrics = ", ".join(str(metric) for metric in chosen_metric)
            st.session_state.pipeline_summary = f"""
                                **Dataset:** {dataset} \n
                                **Input Features:** {inp_feat} \n
                                **Target Feature:** {target_feature_name} \n
                                **Task Type:** {task_type} \n
                                **Split:** {split} \n
                                **Model:** {model_choice} \n
                                **Metric(s):** {metrics}
                                """
            st.write(st.session_state.pipeline_summary)

            # get a confirmation from the user before executing the pipeline
            if st.button(label="Confirm and Train Model"):
                pipeline = Pipeline(chosen_metric, dataset, chosen_model,
                                    input_features, target_feature, split)

                pipeline_return = pipeline.execute()

                st.subheader("Train Metrics")
                for metric, result in pipeline_return['train metrics']:
                    st.write(str(metric), ": ", result)

                st.subheader("Test Metrics")
                for metric, result in pipeline_return['test metrics']:
                    st.write(str(metric), ": ", result)

                st.subheader("Predictions")
                pred = [prediction for prediction in pipeline_return[
                    'predictions']]
                st.table(pred)
                st.session_state.pipeline_ready = True

if st.session_state.pipeline_ready:
    st.header('Saving')
    name_of_pipeline = st.text_input(
        "Give a name to your pipeline:")
    pipeline_version = st.text_input(label='Version')

    if st.button(label='Confirm and Save'):
        pipeline_artifact = Artifact(
            type='pipeline',
            data=st.session_state.pipeline_summary.encode(),
            name=name_of_pipeline,
            version=pipeline_version,
            asset_path='./assets/pipelines')
        automl._registry.register(pipeline_artifact)
        st.success(f"Pipeline '{name_of_pipeline}' version"
                   f" '{pipeline_version}' saved successfully.")
