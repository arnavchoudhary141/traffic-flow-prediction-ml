from pathlib import Path
import pandas as pd

FEATURES = ["road_id", "day_of_week", "hour"]
TARGET = "congestion_level"
REQUIRED = FEATURES + [TARGET]

def load_dataset(path):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df[REQUIRED].copy()
    for col in REQUIRED:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates()
    df = df[df["day_of_week"].between(0, 6)]
    df = df[df["hour"].between(0, 23)]
    df = df[df[TARGET].between(0, 4)]
    df[["road_id", "day_of_week", "hour", TARGET]] = df[
        ["road_id", "day_of_week", "hour", TARGET]
    ].astype(int)
    if len(df) < 25:
        raise ValueError("Need at least 25 valid rows to train and evaluate.")
    if df[TARGET].nunique() < 2:
        raise ValueError("Target must contain at least two congestion classes.")
    return df
