AUTOOP
    TESTS
        - We implemented unit tests for all the AutoOP files to ensure proper functionality.
        The coded pipeline test additionally provides a comprehensive check on how the files work together.

    FUNCTIONAL
        - To avoid redundancy, we do not return a copy of the results in preprocessing.py since it is only
        used in the Pipeline. We handle returning copies there, as it is used in app/Modelling.
        - In detect_feature_types, we return a deepcopy as it is used in the app/Modelling file.

    CORE
        STORAGE
            - We added a NotFoundError for improved clarity and understanding for the user.
        DATABASE
            - No deepcopies are returned as this class is not directly used on the Streamlit app pages.
        ML
            ARTIFACT
                - We included the tags parameter, even though it remains unchanged throughout the program.
                Since all artifacts relate to machine learning, the rest of the essential information is
                provided by other parameters.
                - In the read method, we do not return a deepcopy because we don’t expose it directly
                to the user, except in app/Modelling, but we don’t use the read method there, 
                instead create our own table.
                - A getter was created for the name, as it is exposed on the Streamlit page.
            DATASET
                - No changes required.
            FEATURE
                - We created getters for the two attributes since we might access them on the Streamlit
                app if displaying the features.
                - When we generate a table in app/Modelling to display feature names and types, we return
                a deepcopy. Since strings are immutable, no deepcopies are necessary for their return values.
            METRIC
                - Immutable values are returned without deepcopying.
                - For get_metric, which returns a list, we return a deepcopy.
            PIPELINE
                - A deepcopy is returned from execute() since it is exposed in app/Modelling.
                - Preprocessing was adjusted for better clarity when running the Streamlit app.
                Initially, it did not correctly interpret the target feature, returning only numerical
                values instead of labels.
            MODEL
                - The abstract model class initializes two attributes - type and parameters. Both are
                used across different initializations.
                - Getters are provided: type is returned as-is since it’s immutable, and parameters
                are returned as deepcopies.
                - Getters are implemented as concrete methods, identical across all subclasses, to avoid
                redundancy.
