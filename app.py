from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("student_risk_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    reasons = []

    if request.method == "POST":
        attendance = float(request.form["attendance"])
        assignment = float(request.form["assignment"])
        participation = float(request.form["participation"])

        data = pd.DataFrame(
            [[attendance, assignment, participation]],
            columns=["Attendance", "Assignment", "Participation"]
        )

        result = model.predict(data)[0]

        if attendance < 70:
            reasons.append("Low attendance")
        if assignment < 60:
            reasons.append("Poor assignment submission")
        if participation < 5:
            reasons.append("Weak class participation")

    return render_template("index.html", result=result, reasons=reasons)

if __name__ == "__main__":
    app.run(debug=True)
