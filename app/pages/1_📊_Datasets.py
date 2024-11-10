import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

st.set_page_config(page_title="Datasets", page_icon="📊")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


# create a page introduction
st.write("# Datasets")
write_helper_text("In this section, you can manage the datasets that you want"
                  " to work with.")

# create an outline, and promp the user to upload a dataset
st.subheader("Create")
uploaded_file = st.file_uploader(label="Upload your dataset", type='csv')
file_name = st.text_input("Name the Dataset:")

# save dataset as a Dataset object
if uploaded_file is not None:
    dataset = Dataset.from_dataframe(data=pd.read_csv(uploaded_file),
                                     name=file_name,
                                     asset_path="./assets/dbo")

# create an instance of an Auto ML System, and register the dataset
st.subheader("Save")

if st.button(label="Confirm your chosen dataset"):
    automl = AutoMLSystem.get_instance()
    automl.registry.register(dataset)
    st.write("Dataset confirmed")
