import pandas as pd
import json

# Load the original dataset (adjust path as needed)
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

# Load feature importance (from permutation importance or fallback)
importance_path = 'model_outputs/feature_importance.csv'
importance_df = pd.read_csv(importance_path)

# Select features with nonzero importance
if 'importance' in importance_df.columns:
    selected_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
elif 'coefficient' in importance_df.columns:
    selected_features = importance_df[importance_df['coefficient'] != 0]['feature'].tolist()
else:
    raise ValueError('No importance or coefficient column found in feature importance CSV.')

# Add the target variable (use the original name from the dataset)
if 'Target' in df.columns:
    target_col = 'Target'
else:
    # fallback: try lower/underscore
    target_col = [col for col in df.columns if col.lower() == 'target']
    target_col = target_col[0] if target_col else df.columns[-1]

# Prepare the Looker Studio dataframe
looker_df = df[selected_features + [target_col]]

# Save to CSV for Looker Studio
looker_df.to_csv('model_outputs/looker_studio_data.csv', index=False)
print(f"Exported Looker Studio data with {len(selected_features)} features + target to model_outputs/looker_studio_data.csv") 