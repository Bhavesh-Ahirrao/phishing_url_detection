"""
Dataset Preprocessing and Preparation Module.

Handles loading, validating, cleaning, label normalization, and
splitting of phishing URL datasets.
"""

from __future__ import annotations

import os
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Standard mappings for labels found across common cybersecurity datasets
COMMON_LABEL_MAPPINGS = {
    # Textual representations
    "phishing": 1,
    "malicious": 1,
    "bad": 1,
    "1": 1,
    "1.0": 1,
    1: 1,
    1.0: 1,
    "legitimate": 0,
    "benign": 0,
    "good": 0,
    "safe": 0,
    "0": 0,
    "0.0": 0,
    0: 0,
    0.0: 0,
    # Specific UCI Phishing Websites mapping (-1: Legitimate, 1: Phishing)
    "-1": 0,
    -1: 0,
    -1.0: 0,
}


# Specific mapping for Mendeley LegitPhish dataset where 0 = Phishing and 1 = Legitimate
MENDELEY_LEGITPHISH_MAPPING = {
    0: 1,      # Original 0 (Phishing) -> Normalized 1 (Phishing)
    0.0: 1,
    "0": 1,
    "0.0": 1,
    1: 0,      # Original 1 (Legitimate) -> Normalized 0 (Legitimate)
    1.0: 0,
    "1": 0,
    "1.0": 0,
}


def detect_columns(df: pd.DataFrame, url_col: Optional[str] = None, label_col: Optional[str] = None) -> Tuple[str, str]:
    """
    Detect or validate URL and label column names from DataFrame.
    
    Args:
        df: Input DataFrame.
        url_col: Optional explicit URL column name.
        label_col: Optional explicit label column name.
        
    Returns:
        Tuple of (url_column_name, label_column_name).
    """
    cols_lower = {col.lower().strip(): col for col in df.columns}

    # Detect URL column
    detected_url_col = None
    if url_col and url_col in df.columns:
        detected_url_col = url_col
    else:
        for candidate in ["url", "urls", "raw_url", "url_address", "website", "link", "domain"]:
            if candidate in cols_lower:
                detected_url_col = cols_lower[candidate]
                break

    # Detect Label column
    detected_label_col = None
    if label_col and label_col in df.columns:
        detected_label_col = label_col
    else:
        for candidate in ["classlabel", "class_label", "label", "labels", "class", "target", "result", "status", "type", "is_phishing", "phishing"]:
            if candidate in cols_lower:
                detected_label_col = cols_lower[candidate]
                break

    if not detected_url_col:
        raise ValueError(
            f"Could not automatically detect URL column. Available columns: {list(df.columns)}. "
            "Please specify url_col explicitly."
        )

    if not detected_label_col:
        raise ValueError(
            f"Could not automatically detect Label column. Available columns: {list(df.columns)}. "
            "Please specify label_col explicitly."
        )

    return detected_url_col, detected_label_col


def normalize_labels(series: pd.Series, custom_mapping: Optional[Dict[Any, int]] = None) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Normalize labels into 0 (Legitimate) and 1 (Phishing).
    
    Args:
        series: Raw label series.
        custom_mapping: Optional dictionary mapping raw values to 0 or 1.
        
    Returns:
        Tuple of (normalized_series, mapping_used).
    """
    mapping = custom_mapping if custom_mapping is not None else COMMON_LABEL_MAPPINGS
    
    # Standardize string representations for robust lookup
    def map_val(val: Any) -> int:
        if pd.isna(val):
            raise ValueError("Missing label value encountered.")
        
        # Direct lookup
        if val in mapping:
            return mapping[val]
        
        # Lowercase string lookup
        str_val = str(val).strip().lower()
        if str_val in mapping:
            return mapping[str_val]
        
        # If float-like string: '1.0' -> 1
        try:
            float_val = float(str_val)
            if int(float_val) in mapping:
                return mapping[int(float_val)]
        except ValueError:
            pass

        raise ValueError(
            f"Unrecognized label value: {val!r}. Recognized values are: {list(mapping.keys())}"
        )

    normalized = series.apply(map_val).astype(int)
    # Filter mapping to only observed classes
    unique_raw = series.dropna().unique()
    observed_mapping = {str(k): map_val(k) for k in unique_raw}
    
    return normalized, observed_mapping


def load_and_clean_data(
    file_path: str,
    url_col: Optional[str] = None,
    label_col: Optional[str] = None,
    custom_label_mapping: Optional[Dict[Any, int]] = None,
    dataset_source: Optional[str] = None,
    max_samples: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load dataset from CSV, inspect it, clean missing and duplicates, and normalize labels.
    
    Args:
        file_path: Path to dataset CSV file.
        url_col: Optional URL column name.
        label_col: Optional Label column name.
        custom_label_mapping: Optional mapping for labels.
        dataset_source: 'mendeley', 'uci', or None for auto-detection.
        max_samples: Optional maximum number of stratified samples to retain.
        random_state: Seed for sampling.
        
    Returns:
        Tuple of (cleaned_df, summary_metadata).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    # Read CSV safely with standard encodings
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, low_memory=False, encoding="latin-1")

    initial_records = len(df)
    logger.info(f"Loaded dataset from {file_path} with {initial_records} raw rows and columns: {list(df.columns)}")

    detected_url_col, detected_label_col = detect_columns(df, url_col=url_col, label_col=label_col)

    # Missing value count
    missing_urls = df[detected_url_col].isna().sum()
    missing_labels = df[detected_label_col].isna().sum()

    # Drop missing values
    df_clean = df.dropna(subset=[detected_url_col, detected_label_col]).copy()
    df_clean[detected_url_col] = df_clean[detected_url_col].astype(str).str.strip()

    # Drop empty strings
    df_clean = df_clean[df_clean[detected_url_col] != ""]

    # Count & drop exact duplicates
    duplicates_count = df_clean.duplicated(subset=[detected_url_col]).sum()
    df_clean = df_clean.drop_duplicates(subset=[detected_url_col]).reset_index(drop=True)

    # Determine mapping strategy
    mapping_to_use = custom_label_mapping
    if mapping_to_use is None:
        # Check if dataset is Mendeley LegitPhish (ClassLabel with 0 and 1)
        if dataset_source == "mendeley" or (
            detected_label_col == "ClassLabel" and set(df_clean[detected_label_col].unique()).issubset({0, 1, 0.0, 1.0, "0", "1"})
        ):
            logger.info("Detected Mendeley LegitPhish format: Original 0=Phishing -> 1, Original 1=Legitimate -> 0")
            mapping_to_use = MENDELEY_LEGITPHISH_MAPPING

    # Normalize labels
    normalized_labels, label_mapping = normalize_labels(
        df_clean[detected_label_col], custom_mapping=mapping_to_use
    )
    df_clean["label"] = normalized_labels
    df_clean["url"] = df_clean[detected_url_col]

    # Optional stratified down-sampling for rapid training or demonstration
    if max_samples is not None and len(df_clean) > max_samples:
        logger.info(f"Subsampling {max_samples} records stratifying by class distribution...")
        df_clean, _ = train_test_split(
            df_clean,
            train_size=max_samples,
            stratify=df_clean["label"],
            random_state=random_state,
        )
        df_clean = df_clean.reset_index(drop=True)

    final_records = len(df_clean)
    class_counts = df_clean["label"].value_counts().to_dict()

    summary = {
        "file_path": file_path,
        "initial_records": initial_records,
        "final_records": final_records,
        "missing_urls": int(missing_urls),
        "missing_labels": int(missing_labels),
        "duplicates_removed": int(duplicates_count),
        "url_column": detected_url_col,
        "label_column": detected_label_col,
        "label_mapping": label_mapping,
        "class_distribution": {
            "legitimate (0)": int(class_counts.get(0, 0)),
            "phishing (1)": int(class_counts.get(1, 0)),
        },
    }

    logger.info(
        f"Data cleaned: {final_records} valid unique URLs retained. "
        f"Class distribution: Legitimate (0)={class_counts.get(0, 0)}, Phishing (1)={class_counts.get(1, 0)}"
    )

    return df_clean[["url", "label"]], summary


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into reproducible train and test sets using stratification.
    
    Args:
        df: DataFrame containing 'url' and 'label'.
        test_size: Proportion of test split (default 0.2 = 20%).
        random_state: Seed for reproducibility.
        
    Returns:
        Tuple of (X_train_urls, X_test_urls, y_train, y_test).
    """
    X = df[["url"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
