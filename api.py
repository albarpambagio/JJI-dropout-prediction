import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pycaret.classification import load_model, predict_model
from typing import Union

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
# Reverse mappings for label-to-code
GENDER_MAP_REV = {v: k for k, v in GENDER_MAP.items()}
MARITAL_STATUS_MAP_REV = {v: k for k, v in MARITAL_STATUS_MAP.items()}
BINARY_MAP_REV = {v: k for k, v in BINARY_MAP.items()}

categorical_options = {
    'Gender': GENDER_MAP_REV,
    'Marital_Status': MARITAL_STATUS_MAP_REV,
    'Displaced': BINARY_MAP_REV,
    'Educational_special_needs': BINARY_MAP_REV,
    'Debtor': BINARY_MAP_REV,
    'Tuition_fees_up_to_date': BINARY_MAP_REV,
    'Scholarship_holder': BINARY_MAP_REV,
    'International': BINARY_MAP_REV,
    # Add more mappings as needed for your dataset
}

# Map prediction labels to integer codes
PREDICTION_LABEL_MAP = {
    'Dropout': 0,
    'Enrolled': 1,
    'Graduate': 2
}

def standardize_feature_names(df):
    df = df.copy()
    df.columns = [col.replace(' ', '_').replace('-', '_').replace('/', '_')
                 .replace('(', '').replace(')', '').replace("'", "").strip()
                 for col in df.columns]
    return df

def prepare_input(data: dict):
    # Convert string labels to codes for categorical fields
    for col, mapping in categorical_options.items():
        if col in data and isinstance(data[col], str):
            if data[col] in mapping:
                data[col] = mapping[data[col]]
            else:
                raise HTTPException(status_code=400, detail=f"Invalid value '{data[col]}' for {col}. Allowed: {list(mapping.keys())}")
    df = pd.DataFrame([data])
    df = standardize_feature_names(df)
    for col in required_features:
        if col not in df.columns:
            df[col] = 0
    df = df[required_features]
    return df

class StudentData(BaseModel):
    Marital_Status: Union[int, str] = 1
    Application_mode: int = 1
    Application_order: int = 1
    Course: int = 1
    Daytime_evening_attendance: int = 1
    Previous_qualification: int = 1
    Previous_qualification_grade: float = 120.0
    Nacionality: int = 1
    Mothers_qualification: int = 1
    Fathers_qualification: int = 1
    Mothers_occupation: int = 1
    Fathers_occupation: int = 1
    Admission_grade: float = 120.0
    Displaced: Union[int, str] = 0
    Educational_special_needs: Union[int, str] = 0
    Debtor: Union[int, str] = 0
    Tuition_fees_up_to_date: Union[int, str] = 1
    Gender: Union[int, str] = 1
    Scholarship_holder: Union[int, str] = 0
    Age_at_enrollment: int = 18
    International: Union[int, str] = 0
    Curricular_units_1st_sem_credited: int = 0
    Curricular_units_1st_sem_enrolled: int = 0
    Curricular_units_1st_sem_evaluations: int = 0
    Curricular_units_1st_sem_approved: int = 0
    Curricular_units_1st_sem_grade: float = 10.0
    Curricular_units_1st_sem_without_evaluations: int = 0
    Curricular_units_2nd_sem_credited: int = 0
    Curricular_units_2nd_sem_enrolled: int = 0
    Curricular_units_2nd_sem_evaluations: int = 0
    Curricular_units_2nd_sem_approved: int = 0
    Curricular_units_2nd_sem_grade: float = 10.0
    Curricular_units_2nd_sem_without_evaluations: int = 0
    Unemployment_rate: float = 7.0
    Inflation_rate: float = 1.5
    GDP: float = 2.0

app = FastAPI()

@app.post("/predict")
def predict(data: StudentData):
    try:
        df = prepare_input(data.dict())
        result = predict_model(model, data=df)
        pred = result['prediction_label'].iloc[0]
        score = result['prediction_score'].iloc[0]
        print(f"Raw prediction: {pred}, type: {type(pred)}")
        # Map string label to integer code
        if isinstance(pred, str):
            if pred in PREDICTION_LABEL_MAP:
                prediction_int = PREDICTION_LABEL_MAP[pred]
            else:
                raise HTTPException(status_code=400, detail=f"Unknown prediction label: {pred}")
        else:
            prediction_int = int(pred)
        return {"prediction": prediction_int, "probability": float(score)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 