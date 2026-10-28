import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import DATASETS
from src.data_loader import load_dataset


def verify_dataset(name):
    X, y = load_dataset(name)
    df = pd.read_csv(DATASETS[name]["path"], na_values=["?", "\t?", ""])
    for col in DATASETS[name].get("drop", []):
        if col in df.columns:
            df = df.drop(columns=[col])
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    for col in df.select_dtypes(include=["object"]).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        converted = pd.to_numeric(non_null, errors="coerce")
        if converted.notna().all():
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in DATASETS[name].get("zero_missing", []):
        if col in df.columns:
            df[col] = df[col].replace(0, pd.NA)
    print(f"\n=== {name.upper()} ===")
    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")
    print("class counts:")
    print(df[DATASETS[name]["target"]].value_counts().to_string())
    print("missing values per column:")
    print(df.isna().sum().to_string())


if __name__ == "__main__":
    for name in DATASETS:
        verify_dataset(name)
