"""Train Naive Bayes and SVM models for spam detection."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.utils import (
    compute_metrics,
    load_dataset,
    preprocess_series,
    save_metrics,
    summarize_confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "spam.csv"
MODEL_DIR = PROJECT_ROOT / "model"


def train_models() -> None:
    """Train and compare Naive Bayes and SVM models."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATA_PATH)
    dataset["clean_text"] = preprocess_series(dataset["text"])

    x_train, x_test, y_train, y_test = train_test_split(
        dataset["clean_text"],
        dataset["label"],
        test_size=0.2,
        random_state=42,
        stratify=dataset["label"],
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    nb_model = MultinomialNB()
    nb_model.fit(x_train_vec, y_train)
    nb_pred = nb_model.predict(x_test_vec)

    svm_base = LinearSVC()
    svm_model = CalibratedClassifierCV(svm_base)
    svm_model.fit(x_train_vec, y_train)
    svm_pred = svm_model.predict(x_test_vec)

    metrics = {
        "naive_bayes": compute_metrics(y_test.to_numpy(), nb_pred),
        "svm": compute_metrics(y_test.to_numpy(), svm_pred),
    }

    confusion = {
        "naive_bayes": summarize_confusion_matrix(confusion_matrix(y_test, nb_pred)),
        "svm": summarize_confusion_matrix(confusion_matrix(y_test, svm_pred)),
    }

    reports = {
        "naive_bayes": classification_report(y_test, nb_pred),
        "svm": classification_report(y_test, svm_pred),
    }

    best_model_name = max(metrics, key=lambda name: metrics[name]["f1_score"])
    best_model = nb_model if best_model_name == "naive_bayes" else svm_model

    with (MODEL_DIR / "spam_model.pkl").open("wb") as file:
        pickle.dump(best_model, file)

    with (MODEL_DIR / "vectorizer.pkl").open("wb") as file:
        pickle.dump(vectorizer, file)

    save_metrics(MODEL_DIR / "metrics.json", metrics, confusion, reports)
    (MODEL_DIR / "model_choice.txt").write_text(
        f"Best model based on F1-score: {best_model_name}\n"
    )


if __name__ == "__main__":
    train_models()
