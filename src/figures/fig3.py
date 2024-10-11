import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import shap

np.random.seed(42)

# Define a dictionary that maps model filenames to dataset filenames
model_to_dataset = {
    'xgboost_mashai_67.pkl': 'NhanesPrepandemicSubset.csv',
    'xgboost_mashai_35.pkl': 'NhanesPrepandemicSubset.csv',
    'xgboost_all_mashai_67.pkl': 'NhanesPrepandemicAll.csv',
    'xgboost_all_mashai_35.pkl': 'NhanesPrepandemicAll.csv'
}

# Initialize a list to store the SHAP values for each model
shap_values_list = []

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

    # Calculate SHAP values for the test set
    explainer = shap.TreeExplainer(model)  # You can use TreeExplainer for XGBoost models
    shap_values = explainer.shap_values(X_test)

    # Append the SHAP values to the list
    shap_values_list.append(shap_values)

# Plot summary SHAP plots for each model
for i, model_filename in enumerate(model_to_dataset.keys()):
    shap_values = shap_values_list[i]
    model_name = model_filename.split('.')[0]  # Extract model name without extension

    # Create violin plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f'Summary SHAP Violin Plot - {model_name}')
    plt.savefig(f'summary_shap_violin_{model_name}.png', dpi=300)
    plt.clf()

    # Create bar plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title(f'Summary SHAP Bar Plot - {model_name}')
    plt.savefig(f'summary_shap_bar_{model_name}.png', dpi=300)
    plt.clf()

print("Summary SHAP plots have been created and saved.")
