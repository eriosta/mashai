import pandas as pd
import pickle
import sys

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_and_save(model, data, output_path, patient_index):
    y_pred = model.predict(data)
    y_pred_proba = model.predict_proba(data)[:, 1]

    results_df = pd.DataFrame()
    
    if patient_index in data.columns:
        results_df[patient_index] = data[patient_index]
        
    results_df['Predicted Label'] = y_pred
    results_df['Probability'] = y_pred_proba

    results_df.to_csv(output_path, index=False)

data_file = sys.argv[1]
model_name = sys.argv[2]

patient_index = sys.argv[3] if len(sys.argv) > 3 else None

model = load_model(f'{model_name}.pkl')

data = pd.read_csv(data_file)

predict_and_save(model, data, 'predictions.csv', patient_index)