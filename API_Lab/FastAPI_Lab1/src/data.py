import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "iris_extended.csv"

FEATURE_COLS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

LABEL_COL = "Species"


def load_data():
    """
    Load the merged flower dataset from CSV.

    Returns:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Encoded labels
        label_encoder (LabelEncoder): For decoding predictions later
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = set(FEATURE_COLS + [LABEL_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[FEATURE_COLS].values
    y_raw = df[LABEL_COL].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return X, y, label_encoder


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Stratified train/test split.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
