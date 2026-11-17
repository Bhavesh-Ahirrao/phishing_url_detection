"""
Real-Time URL Prediction Pipeline Module.

Loads saved trained model artifacts, validates raw input URLs,
extracts identical feature vectors, and returns prediction classifications,
calibrated probability scores, and security explanations.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, Optional
import joblib
import pandas as pd
from src.feature_extraction import extract_features_df, normalize_url, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "models"
MODEL_FILE_NAME = "phishing_detector_model.joblib"
METADATA_FILE_NAME = "model_metadata.joblib"


class URLPredictor:
    """
    Predictor class encapsulating model loading, input validation,
    feature extraction, and classification inference.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, MODEL_FILE_NAME)
        self.metadata_path = os.path.join(model_dir, METADATA_FILE_NAME)
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names = FEATURE_NAMES
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Safely load trained model and metadata from disk if available.
        """
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found at {self.model_path}. Predictions will be unavailable.")
            return

        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded trained model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            self.model = None

        if os.path.exists(self.metadata_path):
            try:
                self.metadata = joblib.load(self.metadata_path)
                if "feature_names" in self.metadata:
                    self.feature_names = self.metadata["feature_names"]
                logger.info(f"Loaded metadata from {self.metadata_path}")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")

    @property
    def is_model_loaded(self) -> bool:
        """Check if model is successfully loaded and ready for inference."""
        return self.model is not None

    def validate_input(self, raw_url: Any) -> bool:
        """
        Validate if the input is a non-empty string with plausible URL structure.
        
        Args:
            raw_url: Input URL from user.
            
        Returns:
            True if input is non-empty, False otherwise.
        """
        if raw_url is None:
            return False
        cleaned = normalize_url(raw_url)
        if not cleaned or len(cleaned) < 3:
            return False
        return True

    def predict(self, raw_url: str) -> Dict[str, Any]:
        """
        Perform end-to-end prediction on a user-supplied URL.
        
        Args:
            raw_url: Raw URL string.
            
        Returns:
            Dictionary with prediction results, risk score, extracted features,
            and safety advisories.
        """
        # Step 1: Input Validation
        if not self.validate_input(raw_url):
            return {
                "status": "Invalid Input",
                "label": None,
                "phishing_probability": None,
                "confidence_score": 0.0,
                "risk_level": "Unknown",
                "raw_url": raw_url,
                "message": "The input URL is empty, invalid, or too short to analyze.",
                "features": None,
                "disclaimer": "Invalid input could not be evaluated by the model.",
            }

        cleaned_url = normalize_url(raw_url)

        # Step 2: Check model readiness
        if not self.is_model_loaded:
            return {
                "status": "Model Not Loaded",
                "label": None,
                "phishing_probability": None,
                "confidence_score": 0.0,
                "risk_level": "Unknown",
                "raw_url": cleaned_url,
                "message": (
                    f"No trained model found at {self.model_path}. "
                    "Please run 'python train_model.py' to train and save the model."
                ),
                "features": None,
                "disclaimer": "Predictions require a trained model.",
            }

        # Step 3: Feature Extraction (stable column ordering)
        try:
            feat_df = extract_features_df([cleaned_url])
            # Reindex to exact feature names expected by the model
            feat_df = feat_df[self.feature_names]
            features_dict = feat_df.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Feature extraction failed for URL '{cleaned_url}': {e}")
            return {
                "status": "Extraction Error",
                "label": None,
                "phishing_probability": None,
                "confidence_score": 0.0,
                "risk_level": "Unknown",
                "raw_url": cleaned_url,
                "message": f"Failed to extract features: {str(e)}",
                "features": None,
                "disclaimer": "Feature extraction failed.",
            }

        # Step 4: Model Inference
        try:
            pred_class = int(self.model.predict(feat_df)[0])
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return {
                "status": "Prediction Error",
                "label": None,
                "phishing_probability": None,
                "confidence_score": 0.0,
                "risk_level": "Unknown",
                "raw_url": cleaned_url,
                "message": f"Inference error: {str(e)}",
                "features": features_dict,
                "disclaimer": "Model failed during inference.",
            }

        # Step 5: Probability Estimation
        phishing_prob = None
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(feat_df)[0]
                # Probability of positive class (1 = Phishing)
                if len(probs) == 2:
                    phishing_prob = float(probs[1])
                else:
                    phishing_prob = float(probs[0])
            except Exception:
                phishing_prob = 1.0 if pred_class == 1 else 0.0
        else:
            phishing_prob = 1.0 if pred_class == 1 else 0.0

        # Step 6: Determine Risk Level & Human-Readable Status
        if pred_class == 1:
            status = "Potentially Phishing"
            confidence = phishing_prob
            if phishing_prob >= 0.85:
                risk_level = "High Risk"
            elif phishing_prob >= 0.65:
                risk_level = "Moderate Risk"
            else:
                risk_level = "Suspicious"
            explanation = (
                "The machine learning model detected structural patterns, delimiters, "
                "or keyword anomalies characteristic of deceptive phishing links."
            )
        else:
            status = "Legitimate"
            confidence = 1.0 - phishing_prob
            if confidence >= 0.85:
                risk_level = "Low Risk (Likely Safe)"
            else:
                risk_level = "Low-to-Moderate Risk"
            explanation = (
                "The URL structure conforms to typical patterns observed in legitimate domains. "
                "No strong phishing markers were detected."
            )

        disclaimer = (
            "IMPORTANT SECURITY NOTICE: This prediction is a machine-learning statistical estimate "
            "based solely on URL structural and lexical characteristics. It is NOT a definitive security "
            "guarantee. Note that the presence of HTTPS does not establish that a website is trustworthy, "
            "as many phishing websites also use SSL/TLS encryption. Never enter credentials or sensitive data "
            "on unverified domains."
        )

        return {
            "status": status,
            "label": pred_class,
            "phishing_probability": round(phishing_prob, 4),
            "confidence_score": round(confidence, 4),
            "risk_level": risk_level,
            "raw_url": cleaned_url,
            "message": explanation,
            "features": features_dict,
            "disclaimer": disclaimer,
            "model_name": self.metadata.get("selected_model_name", type(self.model).__name__),
        }
