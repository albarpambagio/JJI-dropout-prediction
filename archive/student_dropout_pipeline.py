# %% [markdown]
"""
Student Dropout Prediction Pipeline
----------------------------------
This notebook implements a robust, production-ready pipeline for predicting student dropout using the UCI dataset and PyCaret. It covers data cleaning, feature engineering, model training, evaluation, prediction, and feature importance extraction, with explanations for each step.
"""
# %%
import warnings
warnings.filterwarnings('ignore')
from IPython.display import display
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import joblib
import shap
import os
import json
os.makedirs('model_outputs', exist_ok=True)

# %% [markdown]
"""
## Data Acquisition & Understanding
We fetch the dataset and display its structure. Understanding the data is the first step in any machine learning workflow.
"""
# %%
# Fetch the dataset
student_dropout = fetch_ucirepo(id=697)  # UCI ID for the dataset

# Use the original DataFrame with the target column for PyCaret
# If the target is not already a column, add it
if hasattr(student_dropout.data, 'original'):
    df = student_dropout.data.original.copy()
else:
    # fallback: merge features and targets
    df = pd.concat([student_dropout.data.features, student_dropout.data.targets], axis=1)

print('Data shape:', df.shape)
print('First 5 rows:')
display(df.head())

# EDA (optional, for exploration only)
print('\nTarget value counts:')
display(df['Target'].value_counts())

# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df.drop('Target', axis=1).corr(), cmap='coolwarm', annot=False)
plt.title('Feature Correlation Heatmap')
plt.show()

# %% [markdown]
"""
## Data Preprocessing - Consistent Feature Names
To avoid errors and ensure reproducibility, we standardize feature names and ensure all required features are present. This is critical for robust pipelines and model deployment.
"""
# %%
def standardize_feature_names(df):
    """Ensure consistent feature naming across entire pipeline"""
    df = df.copy()
    # PyCaret's default name cleaning rules:
    df.columns = [col.replace(' ', '_').replace('-', '_').replace('/', '_')
                 .replace('(', '').replace(')', '').replace("'", "").strip()
                 for col in df.columns]
    return df

# Apply to original data
df_clean = standardize_feature_names(df)

# Verify all expected features are present
required_features = [
    'Marital_Status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification',
    'Previous_qualification_grade', 'Nacionality',
    'Mothers_qualification', 'Fathers_qualification',
    'Mothers_occupation', 'Fathers_occupation', 'Admission_grade',
    'Displaced', 'Educational_special_needs', 'Debtor',
    'Tuition_fees_up_to_date', 'Gender', 'Scholarship_holder',
    'Age_at_enrollment', 'International',
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations',
    'Unemployment_rate', 'Inflation_rate', 'GDP'
]

# Ensure all features are present
missing_features = [f for f in required_features if f not in df_clean.columns]
if missing_features:
    print(f"Warning: {len(missing_features)} features missing. Adding with default values.")
    for f in missing_features:
        df_clean[f] = 0  # Or appropriate default value

# %% [markdown]
"""
## Model Development with Consistent Features (scikit-learn version)
We set up PyCaret with the cleaned data, enabling feature selection, multicollinearity removal, and class imbalance correction. This step prepares the data and environment for robust model training.
"""
# %%
# Prepare data
X = df_clean.drop('Target', axis=1)
y = df_clean['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

final_model = model  # for compatibility with rest of code

# %% [markdown]
"""
## Consistent Evaluation and Visualization
We evaluate the finalized model using confusion matrix, class report, and feature importance plots. This provides insight into model performance and areas for improvement.
"""
# %%
def safe_evaluation(model, X_test, y_test):
    """Evaluate model using confusion matrix, class report, and feature importance plots."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('model_outputs/confusion_matrix.png', dpi=300)
    plt.close()
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv('model_outputs/confusion_matrix.csv', index=False)
    # Classification report
    cr = classification_report(y_test, y_pred, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    # Only plot actual class rows and main metrics
    class_rows = [c for c in cr_df.index if c not in ['accuracy', 'macro avg', 'weighted avg']]
    # Export a plot-matching CSV with only per-class metrics (no accuracy row)
    plot_df = cr_df.loc[class_rows, ['precision', 'recall', 'f1-score']].copy()
    plot_df.to_csv('model_outputs/classification_report_plot.csv')
    # Plot heatmap (only class rows, main metrics)
    plt.figure(figsize=(8, 4))
    sns.heatmap(cr_df.loc[class_rows, ['precision', 'recall', 'f1-score']], annot=True, cmap='viridis')
    plt.title('Classification Report')
    plt.tight_layout()
    plt.savefig('model_outputs/classification_report.png', dpi=300)
    plt.close()
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        importance_df.to_csv('model_outputs/feature_importance_fallback.csv', index=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance_df.head(20), palette='viridis')
        plt.title('Top 20 Most Important Features (Native Importance)')
        plt.tight_layout()
        plt.savefig('model_outputs/feature_importance_fallback_plot.png', dpi=300)
        plt.close()
    print("All evaluations completed successfully")

safe_evaluation(final_model, X_test, y_test)

# %% [markdown]
"""
## Model Tuning & Finalization
We save the finalized model and required features for production use. This ensures reproducibility and enables future predictions on new data.
"""
# %%
# Save the final model in the proper directory
joblib.dump(final_model, 'model_outputs/final_student_dropout_model.joblib')

#%%
# 5. When making predictions, ensure same preprocessing
def predict_with_clean_features(model, data):
    """Ensure consistent feature names and all required features present, then predict."""
    data_clean = standardize_feature_names(data)
    for col in required_features:
        if col not in data_clean.columns and col != 'Target':
            data_clean[col] = 0
    feature_cols = [col for col in required_features if col in data_clean.columns]
    X_pred = data_clean[feature_cols]
    preds = model.predict(X_pred)
    proba = model.predict_proba(X_pred)
    result = data_clean.copy()
    result['prediction_label'] = preds
    result['prediction_score'] = proba.max(axis=1)
    return result

# 6. Usage example: predict on original data (or replace with new data)
predictions = predict_with_clean_features(final_model, df)
display(predictions[['Target', 'prediction_label', 'prediction_score']].head())

#%%
# Save required features for production in the same directory
with open('model_outputs/model_features.json', 'w') as f:
    json.dump(required_features, f)

# After model finalization, export feature importances
try:
    result = permutation_importance(
        final_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )
    used_features = X_test.columns
    importance_df = pd.DataFrame({
        'feature': used_features,
        'importance': result.importances_mean
    }).sort_values('importance', ascending=False)
    importance_df.to_csv('model_outputs/feature_importance.csv', index=False)
    print("Successfully exported permutation importance")
    plt.figure(figsize=(12, 8))
    sns.barplot(
        x='importance',
        y='feature',
        data=importance_df.head(20),
        palette='viridis'
    )
    plt.title('Top 20 Most Important Features (Permutation Importance)')
    plt.tight_layout()
    plt.savefig('model_outputs/feature_importance_plot.png', dpi=300)
    plt.close()
except Exception as e:
    print(f"Failed to calculate feature importance: {str(e)}")
    if hasattr(final_model, 'feature_importances_'):
        try:
            importance_df = pd.DataFrame({
                'feature': used_features[:len(final_model.feature_importances_)],
                'importance': final_model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance_df.to_csv('model_outputs/feature_importance_fallback.csv', index=False)
            print("Exported simple feature importance as fallback")
        except:
            print("Could not export any feature importance metrics")

# %%