import joblib
import pandas as pd

model = joblib.load("student_risk_model.pkl")

print("=== AI-Based Attendance & Behavior Risk Analysis System ===\n")

attendance = float(input("Attendance (%): "))
assignment = float(input("Assignment submission (%): "))
participation = float(input("Participation (1-10): "))

student_data = pd.DataFrame(
    [[attendance, assignment, participation]],
    columns=["Attendance", "Assignment", "Participation"]
)

risk = model.predict(student_data)[0]

print("\nPredicted Risk Level:", risk)

# -------- Explainable AI Logic --------
reasons = []

if attendance < 70:
    reasons.append("Low attendance")
if assignment < 60:
    reasons.append("Poor assignment submission")
if participation < 5:
    reasons.append("Weak class participation")

if reasons:
    print("\nRisk Reasons:")
    for r in reasons:
        print("-", r)

# -------- Teacher Alert System --------
print("\nTeacher Action:")
if risk == "High":
    print("Immediate attention required")
elif risk == "Medium":
    print("Monitor student performance weekly")
else:
    print("No action required")

# -------- Improvement Suggestions --------
print("\nImprovement Suggestions:")
if attendance < 80:
    print("- Attend at least 80% of classes")
if assignment < 70:
    print("- Submit all assignments on time")
if participation < 6:
    print("- Actively participate in class discussions")
