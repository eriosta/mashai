import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import shap

np.random.seed(42)

# Define a dictionary that maps model filenames to dataset filenames
model_to_dataset = {
    # 'xgboost_mashai_67.pkl': 'NhanesPrepandemicSubset.csv',
    'xgboost_mashai_35.pkl': 'NhanesPrepandemicSubset.csv',
}

# Create a directory for SHAP outputs
output_dir = 'shap_outputs'
os.makedirs(output_dir, exist_ok=True)

# List of indices for which you want to generate waterfall plots
indices_to_plot = [1181,4344,1641,5080]  # 11, 00, 10, 01

for model_filename in model_to_dataset.keys():
    # Load the model from the file
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    # Load the corresponding dataset
    dataset_filename = model_to_dataset[model_filename]
    df = pd.read_csv(f'data/processed/{dataset_filename}').drop('Unnamed: 0', axis=1)

    # Determine the target column based on the model filename
    if '67' in model_filename:
        target_column = 'isAtRiskMASH67'
        df = df.drop('isAtRiskMASH35', axis=1)
    elif '35' in model_filename:
        target_column = 'isAtRiskMASH35'
        df = df.drop('isAtRiskMASH67', axis=1)
    else:
        raise ValueError("Invalid model filename. Model should end with '67' or '35'.")

    def stratified_split(df, target, test_size=0.2, random_state=42):
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = stratified_split(df, target_column)
    
    x_test_df = pd.DataFrame(X_test, index=X_test.index)
    
    # Calculate SHAP values and create an Explanation object
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    expected_value = explainer.expected_value

    shap_explanation = shap.Explanation(shap_values, base_values=expected_value, data=X_test)

    # Generate waterfall plots for specified indices
    for original_idx in indices_to_plot:
        if original_idx in x_test_df.index:
            test_idx = x_test_df.index.get_loc(original_idx)
            shap.waterfall_plot(shap_explanation[test_idx])
        else:
            print(f"Original index {original_idx} is not in the test set.")
