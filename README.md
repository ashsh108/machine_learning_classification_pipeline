# machine_learning_classification_pipeline
## Introduction
In my solution, I focused mostly on showcasing my full system design capabilities and tried to include most of the
system components essential to serve a simple RF model via an APi and an optional Streamlit UI to test working. The
design flow diagram is as follows:
<img width="1413" height="1807" alt="ML approach" src="https://github.com/user-attachments/assets/a760fb63-92af-4ba1-aa43-92b400d4a77e" />

There are some conscious decisions I took as follows:
Standard Scikit-Learn models are saved as .pkl (Pickle) files, which are highly vulnerable to arbitrary code execution
if intercepted. I converted the model to ONNX (Open Neural Network Exchange). It is a pure computational graph,
making it mathematically strict, significantly faster during inference, and completely immune to malicious Python code
injection.

I built a Microservice Architecture. I utilised FastAPI (fastapi_app.py) to create a REST endpoint and Streamlit
(streamlit_app.py) for a human-readable dashboard. Both services pull the same validated model from a centralised
registry.

Relying on humans to manually approve and promote models introduces bottlenecks, inconsistencies, and delays. I
implemented MLflow with a strict programmatic accuracy-based threshold (this can be customised based on what is
defined as good for production, for example, precision, recall). If a newly trained model beats the baseline accuracy,
my pipeline automatically tags it with the @production alias. If it fails, it is tagged with the @staged alias

## The Codebase
1. data_manager.py: Dataloader functions.
2. model_pipeline.py: This is the training engine. I designed it to handle data splitting, feature scaling, and Random
Forest training. It packages the pipeline into an ONNX graph and logs it to MLflow.
3. drift_monitor.py: I implemented the Kolmogorov-Smirnov test to compare the distributions of incoming production
data against historical training data (p < 0.05 threshold) to detect statistical data drift.
4. fastapi_app.py: I used Pydantic to strictly validate incoming JSON payloads, converted them to 2D NumPy
matrices to satisfy ONNX's strict typing, and exposed a prediction endpoint.
5. streamlit_app.py: I created a clean UI that allows users to input data and get visual risk assessments instantly.
6. monitoring_job.py: I designed this to be run daily to read the API logs, check for data drift, and raise alerts if
underlying data distributions have shifted.
7. test_mlops.py: I wrote a Pytest suite.
