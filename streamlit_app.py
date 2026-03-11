import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc

st.set_page_config(page_title="Loan Default Predictor", layout="centered")

@st.cache_resource
def load_mlflow_model():
    model_uri = "models:/LoanDefaultModel@production"
    return mlflow.pyfunc.load_model(model_uri)

try:
    model = load_mlflow_model()
    st.sidebar.success("ONNX Production Model loaded successfully!")
except Exception as e:
    st.error("Failed to load the model from MLflow.")
    st.warning("Ensure you have run `python train_pipeline.py` and that `onnxruntime` is installed.")
    st.write("**Detailed Error:**", e)
    st.stop() 

st.title("Loan Default Prediction App")
st.write("Enter the customer's financial details below to assess default risk.")

col1, col2 = st.columns(2)
with col1:
    age_customer = st.number_input("Customer Age", min_value=18, value=40)
    age_credit = st.number_input("Credit History Age (Months)", min_value=0, value=10)
    mortgage = st.number_input("Mortgage Balance (EURO)", min_value=0, value=150000)
    student_debt = st.number_input("Student Debt (EURO)", min_value=0, value=20000)

with col2:
    n_cs_contacts = st.number_input("CS Contacts (Last Month)", min_value=0, value=2)
    duration_cs = st.number_input("CS Call Duration (Mins)", min_value=0, value=15)
    n_transactions = st.number_input("Transactions (Last Month)", min_value=0, value=30)
    volume_month = st.number_input("Transaction Volume (EURO)", min_value=0, value=3000)

if st.button("Predict Default Risk", use_container_width=True):
    input_data = pd.DataFrame([{
        "age_customer": float(age_customer),
        "age_credit": float(age_credit),
        "mortgage": float(mortgage),
        "student_debt": float(student_debt),
        "n_customer_service_contacts": float(n_cs_contacts),
        "duration_customer_service": float(duration_cs),
        "n_transactions_last_month": float(n_transactions),
        "volume_last_month": float(volume_month)
    }]).astype(np.float32) 
    
    try:
        raw_prediction = model.predict(input_data)
        
        if isinstance(raw_prediction, pd.DataFrame):
            prediction = raw_prediction.iloc[0, 0]
        elif isinstance(raw_prediction, np.ndarray):
            prediction = raw_prediction[0]
        elif isinstance(raw_prediction, list):
            prediction = raw_prediction[0]
        else:
            prediction = raw_prediction

        
        st.divider()
        st.write('# Chances of default is',int(round(prediction['output_probability'][0][1],2)*100),'%')

            
    except Exception as e:
        st.error("Prediction Failed.")
        st.write("**Error Details:**", e)