from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import mlflow.pyfunc
import os
from datetime import datetime
import pandas as pd

app = FastAPI(
    title="Loan Default Prediction API",
    description="REST API to predict if a customer will default on their loan using an ONNX model.",
    version="1.0"
)

try:
    model_uri = "models:/LoanDefaultModel@production"
    model = mlflow.pyfunc.load_model(model_uri)
    print("ONNX Production Model loaded successfully into API!")
except Exception as e:
    print(f"Failed to load model. Ensure it is registered in MLflow. Error: {e}")
    model = None

class LoanApplication(BaseModel):
    age_customer: float
    age_credit: float
    mortgage: float
    student_debt: float
    n_customer_service_contacts: float
    duration_customer_service: float
    n_transactions_last_month: float
    volume_last_month: float

def log_prediction(application_data: dict, prediction: int):
    """Saves the incoming request to a CSV for asynchronous drift monitoring."""
    log_file = "data/prediction_logs.csv"
    
    application_data['timestamp'] = datetime.now().isoformat()
    application_data['predicted_default'] = prediction
    
    df = pd.DataFrame([application_data])
    
    if not os.path.isfile(log_file):
        df.to_csv(log_file, index=False)
    else:
        df.to_csv(log_file, mode='a', header=False, index=False)

@app.post("/predict")
def predict_default(application: LoanApplication, background_tasks: BackgroundTasks):
    """
    Endpoint that accepts customer financial data and returns a default prediction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="ML Model is currently unavailable.")
    
    try:
        input_matrix = np.array([[
            application.age_customer,
            application.age_credit,
            application.mortgage,
            application.student_debt,
            application.n_customer_service_contacts,
            application.duration_customer_service,
            application.n_transactions_last_month,
            application.volume_last_month
        ]], dtype=np.float32)
        
        raw_prediction = model.predict(input_matrix)
        
        if isinstance(raw_prediction, np.ndarray) or isinstance(raw_prediction, list):
            prediction = round(raw_prediction['output_probability'][0][1],2)
        else:
            prediction = round(raw_prediction['output_probability'][0][1],2)
            

    
        background_tasks.add_task(log_prediction, application.model_dump(), prediction)
    
        return {
            "prediction_probability": prediction,
            "risk_level": "High Risk" if prediction >0.9 else "Low Risk",
            "message": "Prediction generated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "healthy", "model": "LoanDefaultModel@production"}