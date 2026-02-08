"""Run preprocessing steps and save cleaned dataset."""

from __future__ import annotations

from pathlib import Path

from src.utils import load_dataset, preprocess_series

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "spam.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "spam_cleaned.csv"


def run_preprocessing() -> None:
    """Load raw dataset, preprocess text, and save cleaned CSV."""
    dataset = load_dataset(DATA_PATH)
    dataset["clean_text"] = preprocess_series(dataset["text"])
    dataset.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    run_preprocessing()
