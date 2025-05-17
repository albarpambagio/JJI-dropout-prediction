# Student Dropout Prediction Project

## Business Understanding (Background)
Jaya Jaya Institut is a well-established higher education institution that has been operating since the year 2000. Over the years, it has produced many graduates with excellent reputations in their respective fields. However, the institute also faces a significant challenge: a considerable number of students do not complete their studies and drop out before graduation.

A high dropout rate is a serious concern for any educational institution. It not only affects the reputation and perceived quality of the institute but also represents a loss of potential for both the students and the institution. Early identification and intervention for students at risk of dropping out is therefore crucial to improve student retention, support student success, and maintain the institute's standing.

## Problem Statement
Jaya Jaya Institut seeks to proactively detect students who are at risk of dropping out as early as possible. By accurately identifying these students, the institute aims to provide targeted guidance and support, thereby reducing dropout rates and helping more students successfully complete their education.

This project leverages data-driven approaches and machine learning to build a predictive model that can identify students likely to drop out. The insights from this model will enable the institute to intervene early and offer special counseling or support to those most in need.

## Overview
This project provides a robust, end-to-end pipeline for predicting student dropout, enrollment, and graduation using the UCI Student Dropout and Academic Success dataset. It includes data preprocessing, feature engineering, model training with PyCaret, model evaluation, an interactive API, a user-friendly Streamlit web app, and a ready-to-use dataset for Looker Studio dashboards.

## Data Source
- **Dataset:** [UCI Machine Learning Repository – Predict students dropout and academic success](https://archive.ics.uci.edu/ml/datasets/predict+students+dropout+and+academic+success)
- **UCI ID:** 697
- **Description:** Contains demographic, academic, socioeconomic, and macroeconomic features for students, along with their final status (Dropout, Enrolled, Graduate).
- **Variable details:** See [`variable_list.md`](variable_list.md) for full descriptions and value mappings.

## Workflow

### 1. Exploratory Data Analysis (EDA)
- Explore the dataset, visualize target distribution, feature relationships, and correlations.
- Scripts: `eda_student_dropout.py` (or notebook)

### 2. Data Preprocessing & Feature Engineering
- Standardize feature names, handle missing values, and ensure consistency for modeling and deployment.

### 3. Model Training & Evaluation
- Use PyCaret for automated model comparison, feature selection, and handling class imbalance.
- Evaluate model performance with confusion matrix, class report, and feature importance plots.
- Feature importances are extracted for both model interpretation and to prepare data for Looker Studio.
- Script: `student_dropout_pipeline.py`

### 4. API for Predictions
- **FastAPI** app (`api.py`) exposes a `/predict` endpoint for programmatic predictions.
- Accepts both integer codes and string labels for categorical variables.
- Returns predicted class and probability.

### 5. Streamlit Web App
- **Streamlit** app (`app.py`) provides an interactive web interface for predictions.
- Dropdowns for categorical variables, prefilled numeric fields using medians from the training data.
- Tooltips with plain-language explanations for each variable.

### 6. Looker Studio Integration
- The extracted feature importances are used to select the most relevant variables for visualization.
- Script (`prepare_looker_data.py`) exports a CSV with only the most important features (nonzero importance) and the target variable, ready for upload to Looker Studio for dashboarding and visualization.

## Looker Studio Dashboard

An interactive Looker Studio dashboard has been created to visualize key insights from the student dropout prediction model. The dashboard includes:
- Target distribution (dropout, enrolled, graduate)
- Feature impact and importance
- Subgroup analysis by course, gender, scholarship status, and more
- Numeric feature distributions and trends

You can explore the dashboard here: [Student Dropout Prediction Looker Studio Dashboard](https://lookerstudio.google.com/reporting/79693c0f-f238-4125-b217-da852daf7336)

This dashboard enables stakeholders to interactively explore the most important factors influencing student outcomes and supports data-driven decision making for early intervention and student support.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd JJI-dropout-prediction
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # Or, if using uv:
   uv pip install fastapi uvicorn pydantic streamlit pycaret pandas scikit-learn matplotlib seaborn ucimlrepo numpy
   ```

## Usage

### 1. **Run the Model Pipeline**
   ```bash
   python student_dropout_pipeline.py
   ```
   - Trains the model, evaluates, and saves outputs in `model_outputs/`.

### 2. **Start the API**
   ```bash
   uvicorn api:app --reload
   ```
   - Access the `/predict` endpoint at `http://localhost:8000/predict`.
   - Send a JSON payload with student data to get predictions.

### 3. **Launch the Streamlit App**
   ```bash
   streamlit run app.py
   ```
   - Open the web interface in your browser for interactive predictions.

### 4. **Prepare Data for Looker Studio**
   ```bash
   python prepare_looker_data.py
   ```
   - Exports `model_outputs/looker_studio_data.csv` for dashboarding.

### 5. **Build Looker Studio Dashboard**
   - Import the CSV into Looker Studio.
   - Suggested visualizations: target distribution, feature impact, numeric feature distributions, trends, and drilldowns. See project documentation for layout ideas.

## Conclusion

### Insights
- **Class Distribution:**  
  Nearly half of the students in the dataset graduate (49.9%), while 32.1% drop out and 17.9% remain enrolled. This highlights that dropout is a significant issue, affecting about one-third of students.

- **Academic Preparedness and Performance:**  
  - **Admission Grade:** Students who graduate have the highest mean admission grade (128.8), followed by enrolled (125.5), and dropouts (125.0). This suggests that better academic preparedness at entry is associated with successful completion.
  - **First Semester Grades:** The mean first semester grade for graduates is 12.6, for enrolled students 11.1, and for dropouts only 7.3.  
  - **Second Semester Grades:** A similar pattern is observed: graduates (12.7), enrolled (11.1), dropouts (5.9).  
  - **Interpretation:** Poor academic performance in the first and second semesters is a strong early indicator of dropout risk.

- **Financial and Administrative Factors:**  
  - **Debtor Status:** Dropouts are more likely to have outstanding debts (312 out of 1421 dropouts) compared to graduates (101 out of 2209).
  - **Tuition Fees Up To Date:** 457 dropouts had unpaid tuition, compared to only 29 graduates. Keeping tuition up to date is strongly associated with graduation.
  - **Scholarship Holder:** Only 134 dropouts had scholarships, compared to 835 graduates. Scholarships appear to have a protective effect against dropout.

- **Demographic and Socioeconomic Factors:**  
  - **Gender:** Dropout rates are similar for males (701) and females (720), but more females graduate (1661) than males (548).
  - **Displaced Students:** A substantial number of dropouts (669) are displaced (living away from home), suggesting that displacement may increase risk.
  - **Marital Status:** The majority of dropouts are single (1184), but this mirrors the overall student population; further analysis may be needed to clarify the effect.

- **Other Observations:**  
  - **Curricular Progress:** Dropouts have lower numbers of approved and credited curricular units in both semesters.
  - **Macroeconomic Factors:** While included in the model, unemployment, inflation, and GDP show less direct separation by target, but may still contribute contextually.

### Recommendations
- Implement early warning systems using the predictive model to flag at-risk students as soon as key risk factors are detected.
- Provide targeted academic support and counseling for students with low grades or poor academic progress in the first year.
- Offer financial guidance and, where possible, increase access to scholarships or payment plans for students with outstanding debts.
- Monitor and support students from high-risk demographic or socioeconomic backgrounds, including those who are displaced, have special needs, or are international students.

### Action Items
- Integrate the predictive model into the student information system to enable real-time risk detection and intervention.
- Use the Looker Studio dashboard to regularly review dropout trends and the effectiveness of interventions.
- Conduct follow-up qualitative research (e.g., surveys, interviews) with at-risk students to better understand their challenges and refine support programs.
- Continuously update the model and dashboard with new data to improve accuracy and adapt to changing student populations.

## Project Structure

```
JJI-dropout-prediction/
├── archive/
│   ├── student_dropout_pipeline.py
│   ├── eda_student_dropout.py
│   └── ...
├── src/
│   ├── api.py
│   ├── app.py
│   ├── prepare_looker_data.py
│   └── ...
├── notebooks/
│   ├── student_dropout_pipeline.ipynb
│   ├── eda_student_dropout.ipynb
│   └── ...
├── data/
│   └── ...
├── docs/
│   └── ...
├── variable_list.md
├── requirements.txt
├── README.md
├── .gitignore
├── todo.md
├── logs.log
├── uv.lock
├── pyproject.toml
├── implementation-plan.md
└── ...
```

## Credits
- **Dataset:** UCI Machine Learning Repository
- **Modeling:** PyCaret, scikit-learn
- **Web App:** Streamlit
- **API:** FastAPI
- **Visualization:** Looker Studio

## License
This project is for educational and research purposes. See `LICENSE` for details.
