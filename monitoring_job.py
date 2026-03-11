# monitoring_job.py
import pandas as pd
from drift_monitor import DriftMonitor

class AlertSystem:
    """Simulates sending an alert to an engineering team."""
    @staticmethod
    def send_alert(alert_type, details):
        print("\n" + "="*50)
        print(f"ALERT TRIGGERED: {alert_type.upper()}")
        print(f"Details: {details}")
        print("Action Required: Check MLflow dashboard and consider retraining.")
        print("="*50 + "\n")

def run_daily_drift_check():
    print("Starting Asynchronous Drift Monitoring Job...")
    
    try:
        reference_df = pd.read_csv("data/data_training.csv")
        
        current_df = pd.read_csv("data/prediction_logs.csv")
    except FileNotFoundError:
        print("Logs or training data not found. Skipping monitoring.")
        return

    if len(current_df) < 50:
        print(f"Not enough new data to run statistical drift tests (Found {len(current_df)}, need 50).")
        return

    features_to_monitor = [
        'age_customer', 'age_credit', 'mortgage', 'student_debt', 
        'n_customer_service_contacts', 'duration_customer_service', 
        'n_transactions_last_month', 'volume_last_month'
    ]

    monitor = DriftMonitor(p_value_threshold=0.05)

    data_drift_report = monitor.detect_data_drift(reference_df, current_df, features_to_monitor)
    
    if data_drift_report['dataset_drift']:
        drifted_features = [f for f, details in data_drift_report['feature_details'].items() if details['drift_detected']]
        AlertSystem.send_alert(
            "Data Drift Detected", 
            f"The following features have statistically shifted in production: {drifted_features}"
        )
    else:
        print("Data distributions look stable.")

if __name__ == "__main__":
    run_daily_drift_check()