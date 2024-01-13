import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
from sklearn.utils import resample

# Set a random seed
np.random.seed(42)

# List of model filenames
model_filenames = ['xgboost_mashai_67.pkl', 'xgboost_mashai_35.pkl']

for model_filename in model_filenames:
    # Load the model from the file
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    # Load your data
    df = pd.read_csv('data/processed/NhanesPrepandemicSubset.csv').drop('Unnamed: 0', axis=1)
    
    # Determine the target column based on the model filename
    if '67' in model_filename:
        target_column = 'isAtRiskMASH67'
        df = df.drop('isAtRiskMASH35', axis=1)
    elif '35' in model_filename:
        target_column = 'isAtRiskMASH35'
        df = df.drop('isAtRiskMASH67', axis=1)
    else:
        raise ValueError("Invalid model filename. Model should end with '67' or '35'.")

    # Define a function for bootstrap evaluation
    def bootstrap_evaluate(model, X_test, y_test, n_bootstrap=1000):
        aurocs, accuracies, sensitivities, specificities, ppvs, npvs = [], [], [], [], [], []

        for _ in range(n_bootstrap):
            # Bootstrap resampling
            X_resampled, y_resampled = resample(X_test, y_test, random_state=np.random.randint(1, 1000))

            # Predict probabilities
            y_pred_proba = model.predict_proba(X_resampled)[:, 1]

            # Calculate AUROC
            auroc = roc_auc_score(y_resampled, y_pred_proba)
            aurocs.append(auroc)

            # Calculate accuracy
            y_pred = (y_pred_proba > 0.5).astype(int)
            accuracy = accuracy_score(y_resampled, y_pred)
            accuracies.append(accuracy)

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_resampled, y_pred).ravel()

            # Calculate sensitivity, specificity, ppv, and npv
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)

            sensitivities.append(sensitivity)
            specificities.append(specificity)
            ppvs.append(ppv)
            npvs.append(npv)

        return aurocs, accuracies, sensitivities, specificities, ppvs, npvs

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_column]), df[target_column], test_size=0.2, random_state=None, stratify=df[target_column])

    # Perform bootstrap evaluation
    aurocs, accuracies, sensitivities, specificities, ppvs, npvs = bootstrap_evaluate(model, X_test, y_test)

    # Calculate means and confidence intervals
    def calculate_metrics_ci(metrics_list, confidence_level=95):
        mean = np.mean(metrics_list)
        lower_ci = np.percentile(metrics_list, (100 - confidence_level) / 2)
        upper_ci = np.percentile(metrics_list, confidence_level + (100 - confidence_level) / 2)
        return mean, lower_ci, upper_ci

    # Calculate metrics and CIs
    metrics = {
        'AUROC': calculate_metrics_ci(aurocs),
        'Accuracy': calculate_metrics_ci(accuracies),
        'Sensitivity': calculate_metrics_ci(sensitivities),
        'Specificity': calculate_metrics_ci(specificities),
        'PPV': calculate_metrics_ci(ppvs),
        'NPV': calculate_metrics_ci(npvs)
    }

    # Create a DataFrame and export to CSV
    metrics_df = pd.DataFrame(metrics, index=['Mean', 'Lower 95% CI', 'Upper 95% CI'])
    metrics_df.to_csv(f'metrics_{target_column}.csv')
