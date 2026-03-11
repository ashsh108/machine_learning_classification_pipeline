from data_manager import DataLoader
from model_pipeline import LoanDefaultPipeline

if __name__ == "__main__":
    print("Starting Training Pipeline...")
    
    
    loader = DataLoader()
    train_df = loader.load_data("data_training.csv")
    test_df = loader.load_data("data_inference.csv")
    
    features = [
        'age_customer', 'age_credit', 'mortgage', 'student_debt', 
        'n_customer_service_contacts', 'duration_customer_service', 
        'n_transactions_last_month', 'volume_last_month'
    ]
    

    
    pipeline = LoanDefaultPipeline()
    X_train, y_train = pipeline.prepare_features(train_df)
    
    test_acc = pipeline.train_and_log(
        X_train, y_train)

    print("Training Pipeline Execution Complete!")