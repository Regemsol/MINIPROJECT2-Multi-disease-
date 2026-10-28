import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import DATASETS


def load_dataset(name):
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["path"], na_values=["?", "\t?", ""])

    for col in cfg.get("drop", []):
        if col in df.columns:
            df = df.drop(columns=[col])

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    for col in df.select_dtypes(include=['object', 'string']).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        converted = pd.to_numeric(non_null, errors='coerce')
        if converted.notna().all():
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in cfg.get("zero_missing", []):
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    target = cfg["target"]
    positive = cfg["positive"]
    y = (df[target] == positive).astype(int)
    X = df.drop(columns=[target])
    return X, y
