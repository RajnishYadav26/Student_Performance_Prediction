import pandas as pd
from scipy.io import arff


data, meta = arff.loadarff("data/student-performance.arff")


df = pd.DataFrame(data)

# Decode byte columns
for column in df.columns:
    if df[column].dtype == object:
        df[column] = df[column].str.decode("utf-8")

print("\nDataset Loaded Successfully!\n")

print(df.head())

print("\nShape:", df.shape)

print("\nColumns:\n", df.columns.tolist())