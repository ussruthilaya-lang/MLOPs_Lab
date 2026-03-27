import joblib
import numpy as np
from pathlib import Path


# ---------------- Load artifacts once (module import time) ----------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

MODEL_PATH = MODEL_DIR / "flower_model_v2.pkl"
ENCODER_PATH = MODEL_PATH.with_suffix(".labels.pkl")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        f"Run train_v2.py first."
    )

if not ENCODER_PATH.exists():
    raise FileNotFoundError(
        f"Label encoder not found at {ENCODER_PATH}. "
        f"Run train_v2.py first."
    )

# Load trained model and label encoder
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


def predict_data(X):
    """
    Predict species, confidence, and class probabilities for input data.
    """

    X = np.asarray(X)

    # ---- Input validation ----
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(
            f"Expected input shape (n_samples, 4), got {X.shape}"
        )

    # ---- Predict class index ----
    pred_idx = model.predict(X)[0]

    # ---- Decode species label ----
    species = label_encoder.inverse_transform([pred_idx])[0]

    # ---- Predict probabilities ----
    probs = model.predict_proba(X)[0]

    prob_map = {
        label_encoder.classes_[i]: float(probs[i])
        for i in range(len(probs))
    }

    confidence = float(np.max(probs))

    return {
        "species": species,
        "confidence": confidence,
        "probabilities": prob_map,
    }
