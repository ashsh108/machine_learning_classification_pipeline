import numpy as np
import mlflow
import mlflow.onnx  
from mlflow.client import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from skl2onnx import to_onnx 

class LoanDefaultPipeline:
    def __init__(self, experiment_name="Loan_Default_Prediction"):
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)
        self.model_name = "LoanDefaultModel"
        
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier())
        ])

    def prepare_features(self, df):
        X = df.drop(columns=['default', 'id_customer'], axis=1)
        y = df['default']
        return X, y

    def train_and_log(self, X, y,accuracy_threshold=0.80):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if 'default' in X_test.columns:
            print("Data Leakage detected! Target column is still in X_train.")
            print("Current columns in X_train:", X_train.columns.tolist())
            return
        with mlflow.start_run() as run:
            self.pipeline.fit(X_train, y_train)
            test_preds = self.pipeline.predict(X_test)
            test_acc = accuracy_score(y_test, test_preds)
            

            X_sample = X_train[:1].to_numpy().astype(np.float32)
            onnx_model = to_onnx(self.pipeline, X_sample)
            
            mlflow.log_metric("test_accuracy", test_acc)
            
            signature = infer_signature(X_sample, test_preds)
            model_info = mlflow.onnx.log_model(
                onnx_model=onnx_model, 
                name="loan_model_onnx",
                signature=signature,
                registered_model_name=self.model_name
            )
            
            print(f"Secure ONNX Model registered! Run ID: {run.info.run_id}")
            self._promote_model_if_worthy(model_info.registered_model_version, test_acc, accuracy_threshold)
            return test_acc

    def _promote_model_if_worthy(self, model_version, test_acc, threshold=0.50):
        """Uses MlflowClient to manage model lifecycle using modern Aliases."""
        from mlflow.client import MlflowClient
        client = MlflowClient()
        
        if test_acc >= threshold:
            print(f"Accuracy ({test_acc:.2f}) meets threshold ({threshold}). Tagging as Production...")
            
            client.set_registered_model_alias(
                name=self.model_name, 
                alias="production", 
                version=model_version
            )
            print(f"Version {model_version} is now tagged as @production!")
        else:
            client.set_registered_model_alias(
                name=self.model_name, 
                alias="staging", 
                version=model_version
            )
            print(f"Accuracy ({test_acc:.2f}) below threshold ({threshold}). Model not promoted. Tagging as Staging...")