"""Test logistic regression training for action type prediction."""

from pathlib import Path
import json
import joblib
import pytest
from src.models.train_logreg import train_logreg


def test_train_logreg_produces_model_files(tmp_path):
    """LogReg training creates model, preprocessor, and metadata."""
    data_dir = Path("data/processed")
    model_dir = tmp_path / "models"

    result = train_logreg(data_dir, model_dir)

    assert (model_dir / "v1_logreg_model.joblib").exists()
    assert (model_dir / "v1_logreg_metadata.json").exists()
    assert result["validation_examples"] > 0
    assert 0.0 <= result["logreg"]["accuracy"] <= 1.0
    assert 0.0 <= result["logreg"]["balanced_accuracy"] <= 1.0


def test_logreg_model_is_loadable(tmp_path):
    """Saved LogReg model can be loaded and used for prediction."""
    data_dir = Path("data/processed")
    model_dir = tmp_path / "models"

    train_logreg(data_dir, model_dir)

    model = joblib.load(model_dir / "v1_logreg_model.joblib")
    assert model is not None
    assert hasattr(model, 'predict')
