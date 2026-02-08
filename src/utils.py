"""Utility functions for preprocessing, evaluation, and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PHISHING_KEYWORDS = {
    "password",
    "bank",
    "verify",
    "urgent",
    "click",
    "login",
    "account",
    "security",
    "confirm",
    "ssn",
    "otp",
    "wire",
    "invoice",
    "paypal",
    "reset",
}


def ensure_nltk_data() -> None:
    """Ensure required NLTK resources are available."""
    resources = [
        "tokenizers/punkt",
        "corpora/stopwords",
        "corpora/wordnet",
        "corpora/omw-1.4",
    ]
    for resource in resources:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/")[-1], quiet=True)


def preprocess_text(text: str) -> str:
    """Preprocess input text with lowercasing, tokenization, stopword removal, and lemmatization."""
    ensure_nltk_data()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token.isalpha() and token not in stop_words
    ]
    return " ".join(cleaned)


def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply preprocessing to an entire pandas Series."""
    return series.fillna("").apply(preprocess_text)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load dataset from CSV file with expected columns: label, text."""
    data = pd.read_csv(path)
    if not {"label", "text"}.issubset(data.columns):
        raise ValueError("Dataset must include 'label' and 'text' columns.")
    return data[["label", "text"]]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute core classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="spam"),
        "recall": recall_score(y_true, y_pred, pos_label="spam"),
        "f1_score": f1_score(y_true, y_pred, pos_label="spam"),
    }


def save_metrics(path: Path, metrics: Dict[str, Dict[str, float]],
                 confusion: Dict[str, List[List[int]]],
                 reports: Dict[str, str]) -> None:
    """Save metrics, confusion matrices, and classification reports to JSON."""
    payload = {
        "metrics": metrics,
        "confusion_matrices": confusion,
        "classification_reports": reports,
    }
    path.write_text(json.dumps(payload, indent=2))


def highlight_phishing_keywords(text: str) -> Tuple[str, List[str]]:
    """Highlight phishing keywords and return matched keywords."""
    tokens = text.split()
    matched = {token.strip(".,!?:;\"'()").lower() for token in tokens}
    hits = sorted(PHISHING_KEYWORDS.intersection(matched))
    highlighted = []
    for token in tokens:
        cleaned = token.strip(".,!?:;\"'()").lower()
        if cleaned in PHISHING_KEYWORDS:
            highlighted.append(f"**{token}**")
        else:
            highlighted.append(token)
    return " ".join(highlighted), hits


def summarize_confusion_matrix(cm: np.ndarray) -> List[List[int]]:
    """Convert numpy confusion matrix to a JSON-serializable list."""
    return cm.tolist()
