# ML-program-for-HVAC-airflow-system-
HVAC Airflow Classification using Machine Learning
Page 1: Project Overview and Dataset Description
1. Introduction
In HVAC (Heating, Ventilation, and Air Conditioning) systems, maintaining balanced airflow is essential for energy efficiency and occupant comfort. This project aims to develop a machine learning model that can classify whether the airflow is balanced or unbalanced, based on temperature and pressure inputs.

Balanced airflow ensures consistent indoor temperatures, reduced energy consumption, and better equipment performance. Automating this classification can significantly assist HVAC operators in early detection of inefficiencies or malfunctions.
2. Dataset Description
A synthetic dataset of 50,000 entries was generated for this project. Each data entry represents a real-time reading from an HVAC system and includes the following features:
Column Name	Description
Temperature (°C)	Air temperature inside the duct (-15 to 0 °C)
Pressure (atm)	Internal pressure in the duct (0.98 to 1.48 atm)
Airflow (m³/min)	Volume of airflow (5 to 8 m³/min)
Surrounding Temp (°C)	Ambient room temperature (5 to 25 °C)
Airflow Status	Label - 1 = Balanced, 0 = Unbalanced

Labeling Logic:
- Balanced Airflow (1): Temperature between -10°C and -5°C and Pressure between 1.1 atm and 1.3 atm
- Unbalanced Airflow (0): All other conditions outside the above range.
Page 2: Machine Learning Model and Code Implementation
3. Model Selection
For this binary classification problem, we chose Logistic Regression, a widely used, interpretable algorithm suitable for simple decision-making based on continuous inputs.
4. Code Implementation
Here’s the core Python code used to train and save the model:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_excel('HVAC_Airflow_Data (1).xlsx')

# Select relevant features and target
X = df[['Pressure (atm)', 'Temperature (°C)']]
y = df['Airflow Status']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save the model
joblib.dump(model, 'hvac_airflow_classifier.pkl')

5. Deployment and Testing
Once saved as hvac_airflow_classifier.pkl, the model can be loaded into any Python environment or IoT system and used like this:

model = joblib.load('hvac_airflow_classifier.pkl')

# Predict using new data
sample = pd.DataFrame({'Pressure (atm)': [1.2], 'Temperature (°C)': [-6]})
print("Prediction:", model.predict(sample))  # Output: [1] for balanced

6. Conclusion
This machine learning solution effectively predicts airflow balance based on two environmental parameters. It can be integrated with real-time HVAC monitoring systems, reducing manual checks and improving indoor climate control. The simplicity of the model ensures fast predictions even in embedded environments.
