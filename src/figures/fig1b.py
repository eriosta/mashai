import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score

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
    'xgboost_mashai_67.pkl': 'NhanesPrepandemicSubset.csv',
    'xgboost_mashai_35.pkl': 'NhanesPrepandemicSubset.csv',
    'xgboost_all_mashai_67.pkl': 'NhanesPrepandemicAll.csv',
    'xgboost_all_mashai_35.pkl': 'NhanesPrepandemicAll.csv'
}

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

    # Stratified K-Fold on the test set
    cv = StratifiedKFold(n_splits=5)

    # Set font properties
    plt.rcParams['font.size'] = 14  # Set the font size to 14
    plt.rcParams['font.family'] = 'Arial'

    # Plotting Precision-Recall Curves
    title = f'Precision-Recall with K-fold Cross-Validation - {model_filename}'

    plt.figure(figsize=(7, 6.5))

    for i, (train, test) in enumerate(cv.split(X_test, y_test)):
        y_pred_proba = model.predict_proba(X_test.iloc[test])[:, 1]
        precision, recall, _ = precision_recall_curve(y_test.iloc[test], y_pred_proba)
        avg_precision = average_precision_score(y_test.iloc[test], y_pred_proba)
        plt.plot(recall, precision, lw=2, color=colors_list[i % len(colors_list)], label=f'AP, k-fold {i+1}: {avg_precision:.2f}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")

    # Save the figure as a high-quality TIFF file
    plt.savefig(f'{title}.tiff', format='tiff', dpi=300, bbox_inches='tight')

    plt.show() # You can remove this line or comment it out as the plot is now being saved to a file