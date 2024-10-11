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
    'xgboost_mashai_67.pkl': 'NhanesPrepandemicSubset.csv',
    'xgboost_mashai_35.pkl': 'NhanesPrepandemicSubset.csv',
    # 'xgboost_all_mashai_67.pkl': 'NhanesPrepandemicAll.csv',
    # 'xgboost_all_mashai_35.pkl': 'NhanesPrepandemicAll.csv'
}

# Create a directory for SHAP outputs
output_dir = 'shap_outputs'
os.makedirs(output_dir, exist_ok=True)

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
    
    # Create a DataFrame for X_test
    x_test_df = pd.DataFrame(X_test, index=X_test.index)
    
    # Calculate the base SHAP value
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    base_shap_value = explainer.expected_value

    # Calculate the corresponding average prediction value
    avg_pred_value = np.mean(model.predict_proba(X_test)[:, 1])

    
    # Create a DataFrame for SHAP values and use original feature names
    shap_value_df = pd.DataFrame(shap_values, columns=x_test_df.columns, index=x_test_df.index)

    # Export X_test, y_test, predicted y, and probability of prediction of y to CSV
    output_csv = os.path.join(output_dir, f'{model_filename.split(".")[0]}_results.csv')
    results_df = pd.DataFrame({'y_test': y_test.tolist(),
                               'y_pred': model.predict(X_test).tolist(),
                               'y_prob': model.predict_proba(X_test)[:, 1].tolist()}, index=x_test_df.index)

    # Combine all DataFrames by index
    final_df = pd.concat([x_test_df, results_df, shap_value_df], axis=1)
    
    # Create a column 'type_error' where 1 if y_test 1 and y_pred 0, and 2 if y_test 0 and y_pred 1 else 0
    final_df['type_error'] = np.where((final_df['y_test'] == 1) & (final_df['y_pred'] == 0), 1,
                                    np.where((final_df['y_test'] == 0) & (final_df['y_pred'] == 1), 2, 0))

    final_df.to_csv(output_csv)
    
    # Count the total number for each type of error and correct predictions
    total_type_1_error = (final_df['type_error'] == 1).sum()
    total_type_2_error = (final_df['type_error'] == 2).sum()
    total_correct = (final_df['type_error'] == 0).sum()

    # Creating a DataFrame to hold the values
    summary_df = pd.DataFrame({
        'Base SHAP Value': [base_shap_value],
        'Average Prediction Value': [avg_pred_value],
        'Total Type 1 Error': [total_type_1_error],
        'Total Type 2 Error': [total_type_2_error],
        'Total Correct': [total_correct]
    })

    # Export to CSV
    summary_csv = os.path.join(output_dir, f'{model_filename.split(".")[0]}_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
        
    # Determine the global min and max of SHAP values
    shap_min = min(shap_value_df.min().min(), base_shap_value)
    shap_max = max(shap_value_df.max().max(), base_shap_value)

    # Adjust the range slightly to ensure visibility of the base line
    y_axis_range = max(abs(shap_max - base_shap_value), abs(base_shap_value - shap_min))
    shap_min = base_shap_value - y_axis_range
    shap_max = base_shap_value + y_axis_range

    # Plotting SHAP values
    n_features = len(x_test_df.columns)
    plt.figure(figsize=(n_features * 5, 5))  # Adjust the size as needed

    for i, feature in enumerate(x_test_df.columns):
        ax = plt.subplot(1, n_features, i + 1)

        # Filter and plot correct predictions
        indices = final_df[final_df['type_error'] == 0].index
        plt.scatter(x_test_df.loc[indices, feature], shap_value_df.loc[indices, feature],
                    c='lightgray', alpha=0.8, s=10)

        # Filter and plot false negatives
        indices = final_df[final_df['type_error'] == 1].index
        plt.scatter(x_test_df.loc[indices, feature], shap_value_df.loc[indices, feature],
                    c='#E64B35FF', alpha=0.8, s=30)

        # Filter and plot false positives
        indices = final_df[final_df['type_error'] == 2].index
        plt.scatter(x_test_df.loc[indices, feature], shap_value_df.loc[indices, feature],
                    c='#00A087FF', alpha=0.8, s=30)

        # Add a horizontal line at y=0
        plt.axhline(y=base_shap_value, color='k', linestyle=':')
        
        plt.axhline(y=0, color='gray', linestyle='-')

       # Set the same y-axis limits for each subplot
        plt.ylim(shap_min, shap_max)
        
        plt.rcParams['font.size'] = 14  # Set the font size to 14
        plt.rcParams['font.family'] = 'Arial'
        
        # Remove top and right borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xlabel(feature)
        plt.ylabel('SHAP values')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_filename.split(".")[0]}_shap_plot.png'))
    plt.close()