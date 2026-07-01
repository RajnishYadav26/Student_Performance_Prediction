import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_dataset():

    data, meta = arff.loadarff("data/student-performance.arff")

    df = pd.DataFrame(data)

    # Decode bytes to string
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

    return df


def preprocess_data():

    df = load_dataset()

    print("=" * 60)
    print("Before Cleaning")
    print("=" * 60)

    print(df.shape)

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Remove missing values
    df.dropna(inplace=True)

    # Remove StudentID
    if "StudentID" in df.columns:
        df.drop(columns=["StudentID"], inplace=True)

    print("\nAfter Cleaning")
    print(df.shape)

    # Features
    X = df.drop("GradeClass", axis=1)

    # Target
    y = df["GradeClass"]

    # Train-Test Split
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

    os.makedirs("model", exist_ok=True)

    joblib.dump(scaler, "model/scaler.pkl")

    print("\nData Preprocessing Completed Successfully.")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    X_train, X_test, y_train, y_test = preprocess_data()

    print("\nTraining Data Shape :", X_train.shape)
    print("Testing Data Shape  :", X_test.shape)