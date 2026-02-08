"""Streamlit app for real-time spam detection."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.predict import load_artifacts
from src.utils import highlight_phishing_keywords, preprocess_text

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"


@st.cache_resource
def load_resources() -> tuple[object, object, dict]:
    """Load model artifacts and metrics."""
    model, vectorizer = load_artifacts()
    metrics_path = MODEL_DIR / "metrics.json"
    metrics_data = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return model, vectorizer, metrics_data


def predict_text(model: object, vectorizer: object, text: str) -> tuple[str, float]:
    """Predict spam label and confidence score."""
    cleaned = preprocess_text(text)
    features = vectorizer.transform([cleaned])
    label = model.predict(features)[0]
    confidence = 0.5
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba(features)[0].max())
    return label, confidence


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="Email Spam Filter", page_icon="📧")
    st.title("📧 Email Spam Filter Using Machine Learning")
    st.write(
        "Paste an email message to classify it as spam or not spam."
    )

    model, vectorizer, metrics_data = load_resources()

    text_input = st.text_area(
        "Email text",
        height=200,
        placeholder="Enter or paste the email content here...",
    )

    if st.button("Analyze"):
        if not text_input.strip():
            st.warning("Please enter some email text to analyze.")
        else:
            label, confidence = predict_text(model, vectorizer, text_input)
            highlighted, hits = highlight_phishing_keywords(text_input)
            st.subheader("Prediction")
            st.write(f"**Label:** {label}")
            st.write(f"**Confidence:** {confidence:.2%}")

            if hits:
                st.subheader("Phishing Keyword Highlights")
                st.markdown(highlighted)
                st.caption(f"Detected keywords: {', '.join(hits)}")
            else:
                st.info("No phishing keywords detected in the message.")

    if metrics_data:
        st.subheader("Model Comparison")
        metrics = metrics_data.get("metrics", {})
        if metrics:
            st.json(metrics)


if __name__ == "__main__":
    main()
