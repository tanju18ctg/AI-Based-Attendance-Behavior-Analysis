import joblib
import pandas as pd

# Load trained model
model = joblib.load("student_risk_model.pkl")

print("=== AI-Based Student Risk Predictor ===")
print("Enter student data to predict risk level:\n")

attendance = float(input("Attendance (%): "))
assignment = float(input("Assignment submission (%): "))
participation = float(input("Participation (1-10): "))

# Create DataFrame with feature names
student_data = pd.DataFrame(
    [[attendance, assignment, participation]],
    columns=["Attendance", "Assignment", "Participation"]
)

# Predict
risk = model.predict(student_data)[0]

print("\nPredicted Risk Level:", risk)
