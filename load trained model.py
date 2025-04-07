import joblib

# Load the trained model
model_path = "/content/hvac_airflow_classifier.pkl"
model = joblib.load(model_path)

print("Model loaded successfully!")
