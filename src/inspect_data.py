import pandas as pd

df = pd.read_csv("data/ai4i2020.csv")
print("=== SHAPE ===")
print(df.shape)

print("\n=== COLUMNS & DTYPES ===")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== MACHINE FAILURE DISTRIBUTION ===")
print(df["Machine failure"].value_counts())
print(df["Machine failure"].value_counts(normalize=True) * 100)

print("\n=== FAILURE MODES BREAKDOWN ===")
modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
for m in modes:
    print(f"{m}: {df[m].sum()} ({df[m].sum()/len(df)*100:.2f}%)")

print("\n=== PRODUCT TYPES ===")
print(df["Type"].value_counts())

print("\n=== NUMERICAL SUMMARY ===")
num_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
print(df[num_cols].describe().round(2))
