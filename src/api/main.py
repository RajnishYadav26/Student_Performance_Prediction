from fastapi import FastAPI
from src.api.schema import StudentData
import joblib
import numpy as np

app = FastAPI(
    title="Student Performance Prediction API",
    description="Predict Student Grade Class",
    version="1.0.0"
)

# Load model and scaler
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")


@app.get("/")
def home():
    return {
        "message": "Student Performance Prediction API",
        "status": "Running"
    }


@app.get("/health")
def health():
    return {
        "status": "Healthy"
    }


@app.post("/predict")
def predict(student: StudentData):

    features = np.array([
        [
            student.Age,
            student.Gender,
            student.Ethnicity,
            student.ParentalEducation,
            student.StudyTimeWeekly,
            student.Absences,
            student.Tutoring,
            student.ParentalSupport,
            student.Extracurricular,
            student.Sports,
            student.Music,
            student.Volunteering,
            student.GPA
        ]
    ])

    features = scaler.transform(features)

    prediction = model.predict(features)[0]

    return {
        "Predicted Grade Class": int(prediction)
    }