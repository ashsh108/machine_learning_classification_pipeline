import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from data_manager import DataLoader, generate_csv_files
from drift_monitor import DriftMonitor
from model_pipeline import LoanDefaultPipeline
from fastapi_app import app 

client = TestClient(app)




@pytest.fixture
def sample_dataframe():
    """Provides a controlled, dummy dataframe for testing preprocessing logic."""
    return pd.DataFrame({
        'id_customer': [1, 2, 3],
        'age_customer': [25.0, 45.0, 35.0],
        'age_credit': [2.0, 15.0, 8.0],
        'mortgage': [0.0, 200000.0, 100000.0],
        'student_debt': [25000.0, 0.0, 15000.0],
        'n_customer_service_contacts': [1.0, 0.0, 3.0],
        'duration_customer_service': [5.0, 0.0, 20.0],
        'n_transactions_last_month': [15.0, 40.0, 25.0],
        'volume_last_month': [1000.0, 5000.0, 2500.0],
        'default': [1, 0, 0]  
    })




def test_data_loader_success():
    """Test that the data loader correctly reads the generated CSVs."""
    loader = DataLoader(data_dir="data")
    train_df = loader.load_data("data_inference.csv")
    
    assert not train_df.empty, "Train data should not be empty."
    assert "age_customer" in train_df.columns, "Expected feature missing."
    assert "default" in train_df.columns, "Target column missing from train data."

def test_data_loader_file_not_found():
    """Test that the loader raises the correct error for missing files."""
    loader = DataLoader(data_dir="data")
    with pytest.raises(FileNotFoundError):
        loader.load_data("non_existent_file.csv")



def test_prevent_data_leakage(sample_dataframe):
    """
    CRITICAL MLOPS TEST: Ensures the target and ID columns are 
    strictly removed from the training features (X).
    """
    pipeline = LoanDefaultPipeline(experiment_name="Test_Experiment")
    X, y = pipeline.prepare_features(sample_dataframe)
    
    assert 'id_customer' not in X.columns, "Data Leakage: ID column found in X!"
    assert 'default' not in X.columns, "Data Leakage: Default column found in X!"
    
    assert len(X.columns) == 8, f"Expected 8 features, got {len(X.columns)}"
    

def test_data_drift_detection():
    """Test that the KS-test logic correctly flags statistically significant drift."""
    monitor = DriftMonitor(p_value_threshold=0.05)
    
    np.random.seed(42)
    reference_df = pd.DataFrame({'income': np.random.normal(50000, 10000, 1000)})
    
    current_df_drifted = pd.DataFrame({'income': np.random.normal(100, 3000, 1000)})
    
    current_df_stable = pd.DataFrame({'income': np.random.normal(50000, 10000, 1000)})
    
    report_drift = monitor.detect_data_drift(reference_df, current_df_drifted, ['income'])
    assert report_drift['feature_details']['income']['drift_detected'] == True
    
    report_stable = monitor.detect_data_drift(reference_df, current_df_stable, ['income'])
    assert report_stable['feature_details']['income']['drift_detected'] == False


def test_api_health_check():
    """Test if the FastAPI server boots up correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_predict_validation_error():
    """
    Test that the API rejects bad payloads.
    If we send missing data or strings instead of numbers, FastAPI should block it.
    """
    bad_payload = {
        "age_customer": "twenty-five", 
        "mortgage": 150000.0
    }
    
    response = client.post("/predict", json=bad_payload)
    
    assert response.status_code == 422