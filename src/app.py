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

# Variable descriptions from variable_list.md
variable_descriptions = {
    'Marital_Status': "Student's marital status, which may affect social support and academic persistence. (1=single, 2=married, 3=widower, 4=divorced, 5=facto union, 6=legally separated)",
    'Application_mode': "How the student applied to the university. Different modes may reflect different backgrounds or entry requirements. (See variable_list.md for codes)",
    'Application_order': "Order of preference for the course applied to (0=first choice, 9=last choice). Lower values may indicate higher motivation.",
    'Course': "The specific degree program the student enrolled in. Some courses may have higher or lower dropout rates. (See variable_list.md for codes)",
    'Daytime_evening_attendance': "Whether the student attends classes during the day or evening. May relate to work/life balance. (1=daytime, 0=evening)",
    'Previous_qualification': "Highest level of education completed before enrolling. Indicates academic preparedness. (See variable_list.md for codes)",
    'Previous_qualification_grade': "Grade achieved in previous qualification (0-200). Higher grades may predict academic success.",
    'Nacionality': "Student's nationality. May relate to language, culture, or support networks. (See variable_list.md for codes)",
    'Mothers_qualification': "Mother's highest education level. Parental education can influence student outcomes. (See variable_list.md for codes)",
    'Fathers_qualification': "Father's highest education level. Parental education can influence student outcomes. (See variable_list.md for codes)",
    'Mothers_occupation': "Mother's occupation. Socioeconomic status may affect dropout risk. (See variable_list.md for codes)",
    'Fathers_occupation': "Father's occupation. Socioeconomic status may affect dropout risk. (See variable_list.md for codes)",
    'Admission_grade': "Grade used for university admission (0-200). Higher grades may indicate better academic preparation.",
    'Displaced': "Whether the student is living away from their usual home to attend university. (1=yes, 0=no)",
    'Educational_special_needs': "Whether the student has special educational needs. (1=yes, 0=no)",
    'Debtor': "Whether the student has outstanding debts to the university. Financial stress may increase dropout risk. (1=yes, 0=no)",
    'Tuition_fees_up_to_date': "Whether the student's tuition fees are paid up to date. (1=yes, 0=no)",
    'Gender': "Student's gender. (1=male, 0=female)",
    'Scholarship_holder': "Whether the student receives a scholarship. Financial support may reduce dropout risk. (1=yes, 0=no)",
    'Age_at_enrollment': "Student's age at the time of enrollment. Non-traditional (older) students may face different challenges.",
    'International': "Whether the student is an international student. (1=yes, 0=no)",
    'Curricular_units_1st_sem_credited': "Number of course units credited in the 1st semester. Indicates academic progress.",
    'Curricular_units_1st_sem_enrolled': "Number of course units enrolled in the 1st semester.",
    'Curricular_units_1st_sem_evaluations': "Number of evaluations (exams, assignments) in the 1st semester.",
    'Curricular_units_1st_sem_approved': "Number of course units passed in the 1st semester.",
    'Curricular_units_1st_sem_grade': "Average grade in the 1st semester (0-20).",
    'Curricular_units_1st_sem_without_evaluations': "Number of course units without evaluations in the 1st semester.",
    'Curricular_units_2nd_sem_credited': "Number of course units credited in the 2nd semester.",
    'Curricular_units_2nd_sem_enrolled': "Number of course units enrolled in the 2nd semester.",
    'Curricular_units_2nd_sem_evaluations': "Number of evaluations (exams, assignments) in the 2nd semester.",
    'Curricular_units_2nd_sem_approved': "Number of course units passed in the 2nd semester.",
    'Curricular_units_2nd_sem_grade': "Average grade in the 2nd semester (0-20).",
    'Curricular_units_2nd_sem_without_evaluations': "Number of course units without evaluations in the 2nd semester.",
    'Unemployment_rate': "National unemployment rate (%) at the time of enrollment. Economic context may affect student decisions.",
    'Inflation_rate': "National inflation rate (%) at the time of enrollment.",
    'GDP': "Gross Domestic Product (GDP) at the time of enrollment. Economic context may affect student decisions.",
}

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
        help_text = variable_descriptions.get(col, "")
        if col in categorical_options:
            options = categorical_options[col]
            label_to_value = {v: k for k, v in options.items()}
            default_label = list(options.values())[0]
            selected_label = st.selectbox(col, list(options.values()), index=0, help=help_text)
            input_data[col] = label_to_value[selected_label]
        elif "grade" in col or col in [
            "Unemployment_rate", "Inflation_rate", "GDP",
            "Admission_grade", "Curricular_units_1st_sem_grade", "Curricular_units_2nd_sem_grade", "Previous_qualification_grade"
        ] or True:  # treat all other numerics the same for median
            default = numeric_medians.get(col, 0.0)
            input_data[col] = st.number_input(col, value=default, help=help_text)
        else:
            default = numeric_medians.get(col, 0)
            input_data[col] = st.number_input(col, value=default, help=help_text)
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