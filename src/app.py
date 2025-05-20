import json
import pandas as pd
import joblib
import streamlit as st

# Load model and required features
model = joblib.load('model_outputs/final_student_dropout_model.joblib')
with open('model_outputs/model_features.json', 'r') as f:
    required_features = json.load(f)

# Value mappings
GENDER_MAP = {1: 'male', 0: 'female'}
MARITAL_STATUS_MAP = {
    1: 'single', 2: 'married', 3: 'widower', 4: 'divorced',
    5: 'facto union', 6: 'legally separated'
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
}

# Try to load training data for medians
try:
    from ucimlrepo import fetch_ucirepo
    student_dropout = fetch_ucirepo(id=697)
    if hasattr(student_dropout.data, 'original'):
        df = student_dropout.data.original.copy()
    else:
        df = pd.concat([student_dropout.data.features, student_dropout.data.targets], axis=1)
    
    def standardize_feature_names(df):
        df = df.copy()
        df.columns = [col.replace(' ', '_').replace('-', '_').replace('/', '_')
                     .replace('(', '').replace(')', '').replace("'", "").strip()
                     for col in df.columns]
        return df
    
    df = standardize_feature_names(df)
    numeric_medians = {}
    for col in required_features:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            numeric_medians[col] = float(df[col].median())
except Exception as e:
    numeric_medians = {}
    st.warning(f"Could not load training data for medians: {e}")

# Variable descriptions
variable_descriptions = {
    'Marital_Status': "Student's marital status (1=single, 2=married, 3=widower, 4=divorced, 5=facto union, 6=legally separated)",
    # ... (keep all your existing descriptions) ...
    'GDP': "Gross Domestic Product at enrollment time",
}

def standardize_feature_names(df):
    df = df.copy()
    df.columns = [col.replace(' ', '_').replace('-', '_').replace('/', '_')
                 .replace('(', '').replace(')', '').replace("'", "").strip()
                 for col in df.columns]
    return df

def predict_model(model, data):
    """Custom predict function to match PyCaret's output format"""
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)
    return pd.DataFrame({
        'prediction_label': predictions,
        'prediction_score': probabilities.max(axis=1)
    })

def prepare_input(data: dict):
    df = pd.DataFrame([data])
    df = standardize_feature_names(df)
    for col in required_features:
        if col not in df.columns:
            df[col] = 0
    return df[required_features]

# Streamlit UI
st.title("Student Dropout Prediction")

with st.form("input_form"):
    st.write("Enter student information:")
    input_data = {}
    
    for col in required_features:
        help_text = variable_descriptions.get(col, "")
        
        if col in categorical_options:
            options = categorical_options[col]
            selected_label = st.selectbox(
                col, 
                list(options.values()), 
                index=0, 
                help=help_text
            )
            input_data[col] = list(options.keys())[list(options.values()).index(selected_label)]
        else:
            default = numeric_medians.get(col, 0.0 if "." in col else 0)
            input_data[col] = st.number_input(
                col, 
                value=default, 
                help=help_text,
                step=0.1 if isinstance(default, float) else 1
            )
    
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        df = prepare_input(input_data)
        result = predict_model(model, df)
        pred = result['prediction_label'].iloc[0]
        score = result['prediction_score'].iloc[0]
        
        st.success(f"Prediction: {pred} (Probability: {score:.2%})")
        
        st.write("---")
        st.write("**Raw prediction output:**")
        st.dataframe(result)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")