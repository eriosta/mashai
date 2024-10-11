import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# NPG color palette as a dictionary
color_palette = {
    'E64B35FF': '#E64B35FF',
    '4DBBD5FF': '#4DBBD5FF',
    '00A087FF': '#00A087FF',
    '3C5488FF': '#3C5488FF',
    'F39B7FFF': '#F39B7FFF',
    '8491B4FF': '#8491B4FF',
    '91D1C2FF': '#91D1C2FF',
    'DC0000FF': '#DC0000FF',
    '7E6148FF': '#7E6148FF',
    'B09C85FF': '#B09C85FF'
}

# Create a list of colors from the palette
colors_list = list(color_palette.values())

# Define a dictionary that maps model filenames to dataset filenames
model_to_dataset = {
    'xgboost_mashai_67.pkl': 'NhanesPrepandemicSubset.csv', # XGB MASLD N=5, FAST >= 0.67
    'xgboost_mashai_35.pkl': 'NhanesPrepandemicSubset.csv', # XGB MASLD N=5, FAST >= 0.35
    'xgboost_all_mashai_67.pkl': 'NhanesPrepandemicAll.csv', # XGB MASLD N=127, FAST >= 0.67
    'xgboost_all_mashai_35.pkl': 'NhanesPrepandemicAll.csv' # XGB MASLD N=127, FAST >= 0.35
}

def stratified_split(df, target, test_size=0.2, val_size=0.1, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    # Adjust val_size to account for the initial split
    val_size_adjusted = val_size / (1 - test_size)

    # Further split to create the training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Set font properties
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Arial'

# Plotting ROC Curves
plt.figure(figsize=(10, 8))

model_names = {
    'xgboost_mashai_67.pkl': r'XGB MASLD $N_{var}=5$, FAST $\geq$ 0.67',
    'xgboost_mashai_35.pkl': r'XGB MASLD $N_{var}=5$, FAST $\geq$ 0.35',
    'xgboost_all_mashai_67.pkl': r'XGB MASLD $N_{var}=127$, FAST $\geq$ 0.67',
    'xgboost_all_mashai_35.pkl': r'XGB MASLD $N_{var}=127$, FAST $\geq$ 0.35'
}

for i, (model_filename, dataset_filename) in enumerate(model_to_dataset.items()):
    # Load the model from the file
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)

    # Load the corresponding dataset
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

    # Prepare the data
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df, target_column)

    # Generate ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, color=colors_list[i % len(colors_list)], label=f'{model_names[model_filename]}: AUROC = {roc_auc:.2f}')

# Add details to the plot
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('Comparison of AUROC Curves')
plt.legend(loc="lower right")

# Save the figure
plt.savefig('AUROC_Comparison.tiff', format='tiff', dpi=300, bbox_inches='tight')
plt.show()