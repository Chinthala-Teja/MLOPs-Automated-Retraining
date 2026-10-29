
from pathlib import Path
import pandas as pd  # type: ignore[reportMissingImports]

DATA_PATH = Path("data/elec/raw/elec.csv")

df = pd.read_csv(DATA_PATH)

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- FIRST 10 ROWS ---")
print(df.head(10).to_string())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- TARGET DISTRIBUTION ---")
if "class" in df.columns:
    print(df["class"].value_counts(dropna=False))
else:
    print("Target column not identified yet.")

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())
