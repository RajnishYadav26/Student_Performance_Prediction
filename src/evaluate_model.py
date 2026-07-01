import os
import joblib
import pandas as pd

from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

# -------------------------
# Load Dataset
# -------------------------

data, meta = arff.loadarff("data/student-performance.arff")

df = pd.DataFrame(data)

for column in df.columns:
    if df[column].dtype == object:
        df[column] = df[column].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

# -------------------------
# Cleaning
# -------------------------

df.drop_duplicates(inplace=True)

df.dropna(inplace=True)

df.drop(columns=["StudentID"], inplace=True)

X = df.drop("GradeClass", axis=1)

y = df["GradeClass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# -------------------------
# Load Best Model
# -------------------------

model = joblib.load("model/best_model.pkl")

predictions = model.predict(X_test)

# -------------------------
# Classification Report
# -------------------------

report = classification_report(
    y_test,
    predictions
)

print(report)

os.makedirs("reports", exist_ok=True)

with open(
    "reports/classification_report.txt",
    "w"
) as file:

    file.write(report)

# -------------------------
# Confusion Matrix
# -------------------------

cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.savefig(
    "reports/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -------------------------
# Feature Importance
# -------------------------

if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": X.columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    importance.plot.bar(
        x="Feature",
        y="Importance",
        figsize=(10,6)
    )

    plt.tight_layout()

    plt.savefig(
        "reports/feature_importance.png",
        dpi=300
    )

    plt.close()

print("\nEvaluation Completed Successfully")