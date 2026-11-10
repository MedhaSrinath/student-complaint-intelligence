
import pandas as pd

df = pd.read_csv("dataset/complaints.csv")

print("\n========== DATASET INFORMATION ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nDataset Description:")
print(df.describe(include="all"))

