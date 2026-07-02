#  Student Performance Prediction

## Overview

This project predicts student performance using Machine Learning.

The application provides:

- Student Performance Prediction
- Interactive Streamlit Dashboard
- FastAPI REST API
- Data Preprocessing Pipeline
- Model Evaluation
- Logging
- Exception Handling

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- FastAPI
- Streamlit
- Joblib

## Project Structure

Student_Performance_Prediction/
│
├── app/
├── api/
├── config/
├── data/
├── model/
├── reports/
├── src/
├── logs/
├── requirements.txt
├── README.md
└── .gitignore

### Install

pip install -r requirements.txt

### Train

python src/train_model.py

### API

uvicorn api.main:app --reload

### Streamlit

streamlit run app/app.py


## Model

Algorithm:
CatBoost Classifier

Evaluation:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
