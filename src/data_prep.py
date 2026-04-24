"""Data preparation utilities for the Gambia malaria project."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REQUIRED_COLUMNS = ["x", "y", "pos", "age", "netuse", "treated", "green", "phc"]
COVARIATE_COLUMNS = ["age", "netuse", "treated", "green", "phc"]

def load_gambia_data(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    # Ensure required columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[REQUIRED_COLUMNS].dropna().copy()

    # Rename for modeling clarity
    df = df.rename(columns={
        "pos": "malaria",
        "netuse": "bed_net"
    })

    df["malaria"] = df["malaria"].astype(int)

    # Create village from coordinates (CRITICAL FIX)
    df["village_id"] = df.groupby(["x", "y"]).ngroup()

    return df


def make_model_matrices(df):
    scaler = StandardScaler()

    X = scaler.fit_transform(df[["age", "bed_net", "treated", "green", "phc"]])
    y = df["malaria"].to_numpy()
    village = df["village_id"].to_numpy()

    return X, y, village, ["age", "bed_net", "treated", "green", "phc"]