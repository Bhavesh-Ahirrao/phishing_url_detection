"""
Automated Unit Tests for Prediction Pipeline and Preprocessing Validation.

Tests model loading, inference response formatting, missing model handling,
input validation, and dataset label normalization safeguards.
"""

import os
import pytest
import pandas as pd
from src.prediction import URLPredictor
from src.preprocessing import normalize_labels, detect_columns, load_and_clean_data


def test_prediction_output_schema():
    """Verify that predictor returns a comprehensive dictionary with all required keys."""
    predictor = URLPredictor(model_dir="models")
    res = predictor.predict("https://www.wikipedia.org")

    assert isinstance(res, dict)
    required_keys = [
        "status",
        "label",
        "phishing_probability",
        "confidence_score",
        "risk_level",
        "raw_url",
        "message",
        "features",
        "disclaimer",
    ]
    for key in required_keys:
        assert key in res, f"Missing key '{key}' in prediction response."

    assert res["status"] in ["Legitimate", "Potentially Phishing", "Invalid Input"]
    assert "HTTPS" in res["disclaimer"]  # Confirms disclaimer includes HTTPS caveat


def test_invalid_and_empty_url_prediction():
    """Predictor must handle empty, None, and very short inputs safely."""
    predictor = URLPredictor(model_dir="models")

    for bad_input in ["", "  ", None, "ab"]:
        res = predictor.predict(bad_input)
        assert res["status"] == "Invalid Input"
        assert res["label"] is None
        assert res["phishing_probability"] is None


def test_missing_model_directory_graceful_handling(tmp_path):
    """Predictor must not crash when model files are absent; it should report 'Model Not Loaded'."""
    empty_dir = str(tmp_path / "empty_models")
    os.makedirs(empty_dir, exist_ok=True)

    predictor = URLPredictor(model_dir=empty_dir)
    assert not predictor.is_model_loaded

    res = predictor.predict("https://example.com")
    assert res["status"] == "Model Not Loaded"
    assert "train_model.py" in res["message"]


def test_phishing_indicator_prediction():
    """Ensure a suspicious URL with multiple phishing lures is evaluated as Potentially Phishing."""
    predictor = URLPredictor(model_dir="models")
    if not predictor.is_model_loaded:
        pytest.skip("Model not yet trained, skipping inference check.")

    phish_url = "http://192.168.1.1:8888/login-verify-account-update/banking/password.php"
    res = predictor.predict(phish_url)

    assert res["status"] == "Potentially Phishing"
    assert res["label"] == 1
    assert res["phishing_probability"] >= 0.5


def test_label_normalization_standard():
    """Validate label normalization on common cyber label strings."""
    series = pd.Series(["phishing", "legitimate", "malicious", "benign", "1", "0"])
    norm, mapping = normalize_labels(series)

    assert list(norm) == [1, 0, 1, 0, 1, 0]
    assert mapping["phishing"] == 1
    assert mapping["legitimate"] == 0


def test_label_normalization_unsupported_encoding():
    """Unsupported label encodings must raise ValueError rather than produce silent errors."""
    series = pd.Series(["unknown_category_xyz", "legitimate"])
    with pytest.raises(ValueError, match="Unrecognized label value"):
        normalize_labels(series)


def test_detect_columns_unsupported_structure():
    """Detect columns must raise descriptive ValueError if URL or Label column is missing."""
    df_missing_url = pd.DataFrame({"random_col_a": [1, 2], "label": [0, 1]})
    with pytest.raises(ValueError, match="Could not automatically detect URL column"):
        detect_columns(df_missing_url)

    df_missing_label = pd.DataFrame({"url": ["http://a.com", "http://b.com"], "random_col_b": [1, 2]})
    with pytest.raises(ValueError, match="Could not automatically detect Label column"):
        detect_columns(df_missing_label)


def test_load_and_clean_data_missing_file():
    """Ensure load_and_clean_data raises FileNotFoundError for non-existent path."""
    with pytest.raises(FileNotFoundError):
        load_and_clean_data("data/non_existent_dataset_file.csv")
