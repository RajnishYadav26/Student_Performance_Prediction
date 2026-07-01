import os
import joblib
import pandas as pd

from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from logger import logger
logger.info("Training started")

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# -----------------------------
# Load Dataset
# -----------------------------

data, meta = arff.loadarff("data/student-performance.arff")

df = pd.DataFrame(data)

# Decode bytes
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )


# -----------------------------
# Data Cleaning
# -----------------------------

df.drop_duplicates(inplace=True)

df.dropna(inplace=True)

df.drop(columns=["StudentID"], inplace=True)


# -----------------------------
# Split Data
# -----------------------------

X = df.drop("GradeClass", axis=1)

y = df["GradeClass"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# -----------------------------
# Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

os.makedirs("model", exist_ok=True)

joblib.dump(scaler, "model/scaler.pkl")


# -----------------------------
# Models
# -----------------------------

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(random_state=42),

    "XGBoost": XGBClassifier(
        eval_metric="mlogloss",
        random_state=42
    ),

    "LightGBM": LGBMClassifier(random_state=42),

    "CatBoost": CatBoostClassifier(
        verbose=False,
        random_state=42
    ),
}


results = []

best_accuracy = 0

best_model = None

best_model_name = ""


print("=" * 70)
print("Training Models")
print("=" * 70)


for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    results.append(
        [
            name,
            accuracy,
            precision,
            recall,
            f1,
        ]
    )

    print(f"\n{name}")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name


joblib.dump(best_model, "model/best_model.pkl")
logger.info("Best model saved successfully")



print("\n" + "=" * 70)

print("Best Model :", best_model_name)

print("Accuracy   :", round(best_accuracy, 4))

print("=" * 70)


results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ],
)

print("\n")

print(results_df.sort_values(by="Accuracy", ascending=False))


