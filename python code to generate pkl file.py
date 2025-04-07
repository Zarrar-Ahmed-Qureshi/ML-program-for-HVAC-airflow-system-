# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset (Ensure you upload it first)
file_path = "/content/HVAC_Airflow_Data.xlsx"
df = pd.read_excel(file_path)

# Select relevant features and target variable
features = ["Pressure (atm)", "Temperature (°C)"]
target = "Classification"

# Encode the target variable (Balanced = 1, Unbalanced = 0)
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save the trained model
model_path = "/content/hvac_airflow_classifier.pkl"
joblib.dump(model, model_path)

# Print results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Model saved to: {model_path}")
