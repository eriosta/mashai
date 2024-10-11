import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
from streamlit_shap import st_shap

# Set the page layout to wide
st.set_page_config(layout="wide")

st.sidebar.title("MASLD AI")

st.sidebar.markdown(
    'Read our paper here: [Njei et al. (2024). Scientific Reports.](https://www.nature.com/articles/s41598-024-59183-4)'
)


def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, data):
    y_pred = model.predict(data)
    y_pred_proba = model.predict_proba(data)[:, 1]
    return y_pred, y_pred_proba


st.sidebar.header("Enter Data")

model_features = ['ALANINE AMINOTRANSFERASE (ALT) (U/L)', 
                  'GAMMA GLUTAMYL TRANSFERASE (GGT) (IU/L)', 
                  'PLATELET COUNT (1000 CELLS/UL)', 
                  'AGE IN YEARS AT SCREENING', 
                  'BODY MASS INDEX (KG/M**2)']

display_labels = ['ALT (U/L)', 'GGT (U/L)', 'Platelets (1000 cells/µL)', 'Age (years)', 'BMI (kg/m²)']

data = {}
for feature, label in zip(model_features, display_labels):
    data[feature] = st.sidebar.number_input(f'{label}', min_value=0.0)

data_df = pd.DataFrame([data])

if st.sidebar.button('Run Model'):
    with st.spinner('Running the model...'):
        model = load_model('xgboost_mashai_35.pkl')
        y_pred, y_pred_proba = predict(model, data_df)
        
        # Calculate SHAP values
        explainer = shap.Explainer(model)
        shap_values = explainer(data_df)
        
        # Rename features in SHAP values to display shorter feature names
        shap_values.feature_names = display_labels
        
        st.toast('Model ran successfully! 🎉')

    predicted_label = "Likely High-Risk MASH" if y_pred[0] == 1 else "Unlikely High-Risk MASH"

    # st.subheader("Model Predictions")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Prediction", value=predicted_label)
    with col2:
        st.metric(label="Probability", value=f"{y_pred_proba[0]:.2f}")

    # Display SHAP waterfall plot with high quality image
    fig = shap.plots.waterfall(shap_values[0], show=False)
    fig.savefig("shap_waterfall.png", dpi=500, bbox_inches='tight')
    st.image("shap_waterfall.png", caption="SHAP Explanations", use_column_width=True)