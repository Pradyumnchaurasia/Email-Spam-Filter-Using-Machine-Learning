"""Load trained model and generate predictions for new text."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

from src.utils import preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"


def load_artifacts() -> Tuple[object, object]:
    """Load model and vectorizer artifacts."""
    with (MODEL_DIR / "spam_model.pkl").open("rb") as file:
        model = pickle.load(file)
    with (MODEL_DIR / "vectorizer.pkl").open("rb") as file:
        vectorizer = pickle.load(file)
    return model, vectorizer


def predict(text: str) -> Tuple[str, float]:
    """Predict spam label and return label with confidence score."""
    model, vectorizer = load_artifacts()
    cleaned = preprocess_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        confidence = float(np.max(proba))
    else:
        confidence = 0.5

    return prediction, confidence


if __name__ == "__main__":
    sample_text = "Congratulations! You have won a prize. Click to claim now."
    label, score = predict(sample_text)
    print(f"Prediction: {label} (confidence: {score:.2f})")
