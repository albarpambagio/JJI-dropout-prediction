import json
import pandas as pd
import streamlit as st
from pycaret.classification import load_model, predict_model

# Load model and required features
model = load_model('model_outputs/final_student_dropout_model')
with open('model_outputs/model_features.json', 'r') as f:
    required_features = json.load(f)

# Value mappings (update as needed based on your variable_list.md)
GENDER_MAP = {1: 'male', 0: 'female'}
MARITAL_STATUS_MAP = {
    1: 'single', 2: 'married', 3: 'widower', 4: 'divorced', 5: 'facto union', 6: 'legally separated'
}
BINARY_MAP = {0: 'No', 1: 'Yes'}

# Define which variables are categorical and their options
categorical_options = {
    'Gender': GENDER_MAP,
    'Marital_Status': MARITAL_STATUS_MAP,
    'Displaced': BINARY_MAP,
    'Educational_special_needs': BINARY_MAP,
    'Debtor': BINARY_MAP,
    'Tuition_fees_up_to_date': BINARY_MAP,
    'Scholarship_holder': BINARY_MAP,
    'International': BINARY_MAP,
    # Add more mappings as needed for your dataset
}

# --- Calculate medians from the training data for numeric defaults ---
# Try to load the same data used for training
try:
    # You may need to adjust this path if your data is elsewhere
    from ucimlrepo import fetch_ucirepo
    student_dropout = fetch_ucirepo(id=697)
    if hasattr(student_dropout.data, 'original'):
        df = student_dropout.data.original.copy()
    else:
        df = pd.concat([student_dropout.data.features, student_dropout.data.targets], axis=1)
    # Standardize feature names
    def standardize_feature_names(df):
        df = df.copy()
        df.columns = [col.replace(' ', '_').replace('-', '_').replace('/', '_')
                     .replace('(', '').replace(')', '').replace("'", "").strip()
                     for col in df.columns]
        return df
    df = standardize_feature_names(df)
    # Calculate medians for all numeric columns in required_features
    numeric_medians = {}
    for col in required_features:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            numeric_medians[col] = float(df[col].median())
except Exception as e:
    numeric_medians = {}
    st.warning(f"Could not load training data for medians: {e}")

# ---------------------------------------------------------------

def standardize_feature_names(df):
    df = df.copy()
    df.columns = [col.replace(' ', '_').replace('-', '_').replace('/', '_')
                 .replace('(', '').replace(')', '').replace("'", "").strip()
                 for col in df.columns]
    return df

def prepare_input(data: dict):
    df = pd.DataFrame([data])
    df = standardize_feature_names(df)
    for col in required_features:
        if col not in df.columns:
            df[col] = 0
    df = df[required_features]
    return df

st.title("Student Dropout Prediction")

with st.form("input_form"):
    st.write("Enter student information:")
    input_data = {}
    for col in required_features:
        if col in categorical_options:
            options = categorical_options[col]
            label_to_value = {v: k for k, v in options.items()}
            default_label = list(options.values())[0]
            selected_label = st.selectbox(col, list(options.values()), index=0)
            input_data[col] = label_to_value[selected_label]
        elif "grade" in col or col in [
            "Unemployment_rate", "Inflation_rate", "GDP",
            "Admission_grade", "Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_grade", "Previous_qualification_grade"
        ] or True:  # treat all other numerics the same for median
            default = numeric_medians.get(col, 0.0)
            input_data[col] = st.number_input(col, value=default)
        else:
            default = numeric_medians.get(col, 0)
            input_data[col] = st.number_input(col, value=default)
    submitted = st.form_submit_button("Predict")

if submitted:
    df = prepare_input(input_data)
    result = predict_model(model, data=df)
    pred = result['prediction_label'].iloc[0]
    score = result['prediction_score'].iloc[0]
    st.success(f"Prediction: {pred} (Probability: {score:.2f})")

    st.write("---")
    st.write("**Raw prediction output:**")
    st.dataframe(result) 