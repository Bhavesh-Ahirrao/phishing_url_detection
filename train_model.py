"""
Model Training and Evaluation Pipeline.

Trains and compares supervised machine learning models:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier (HistGradientBoosting)

Evaluates performance on held-out test data, selects the best model
by F1-score, persists artifacts with Joblib, and generates evaluation
reports and charts.
"""

from __future__ import annotations

import os
import sys
import argparse
import datetime
import logging
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.feature_extraction import extract_features_df, FEATURE_NAMES
from src.preprocessing import load_and_clean_data, split_data
from src.evaluation import (
    compute_metrics,
    plot_model_comparison,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_roc_curve,
    save_evaluation_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def train_and_evaluate_models(
    dataset_path: str = "data/mendeley_dataset.csv",
    url_col: str | None = None,
    label_col: str | None = None,
    max_samples: int | None = 25000,
    test_size: float = 0.2,
    random_state: int = 42,
    output_model_dir: str = "models",
    output_reports_dir: str = "reports",
) -> Dict[str, Any]:
    """
    Execute end-to-end model training, evaluation, comparison, and persistence.
    """
    logger.info("=" * 70)
    logger.info("STARTING INTELLIGENT PHISHING URL DETECTION TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Load and Clean Dataset
    df_clean, dataset_summary = load_and_clean_data(
        file_path=dataset_path,
        url_col=url_col,
        label_col=label_col,
        max_samples=max_samples,
        random_state=random_state,
    )

    logger.info(f"Dataset summary: {dataset_summary['final_records']} records. "
                f"Distribution: {dataset_summary['class_distribution']}")

    # 2. Extract Lexical & Structural URL Features
    logger.info("Extracting URL features using domain & lexical analyzer...")
    X_features = extract_features_df(df_clean["url"])
    y = df_clean["label"]

    logger.info(f"Feature matrix shape: {X_features.shape}")
    logger.info(f"Feature names ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    # 3. Stratified Train / Test Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Training set: {X_train.shape[0]} samples, Held-out test set: {X_test.shape[0]} samples")

    # 4. Define Candidate Classifiers
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight="balanced",
                solver="lbfgs",
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=14,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=random_state,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=16,
            min_samples_split=6,
            min_samples_leaf=3,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=10,
            min_samples_leaf=10,
            random_state=random_state,
        ),
    }

    # 5. Fit Models & Compute Evaluation Metrics
    eval_results: Dict[str, Any] = {}
    comparison_rows = []
    trained_models = {}

    for name, clf in models.items():
        logger.info(f"\n--- Training {name} ---")
        clf.fit(X_train, y_train)
        trained_models[name] = clf

        # Predictions on held-out test set
        y_pred = clf.predict(X_test)
        
        # Probabilities if available
        y_prob = None
        if hasattr(clf, "predict_proba"):
            try:
                y_prob = clf.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None

        metrics = compute_metrics(y_test, y_pred, y_prob)
        eval_results[name] = metrics

        logger.info(
            f"[{name}] Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1-Score: {metrics['f1_score']:.4f} | "
            f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}"
        )

        comparison_rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"],
            "ROC-AUC": metrics.get("roc_auc", 0.0),
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # 6. Model Selection (Selecting based on highest F1-Score, balancing precision and recall)
    # Phishing detection requires both high detection rate and manageable false alarms
    best_row = comparison_df.sort_values(by=["F1-Score", "Recall"], ascending=False).iloc[0]
    selected_model_name = best_row["Model"]
    selected_model = trained_models[selected_model_name]
    logger.info("=" * 70)
    logger.info(f"SELECTED BEST MODEL: {selected_model_name} (F1-Score: {best_row['F1-Score']:.4f})")
    logger.info("=" * 70)

    # 7. Generate and Save Visualizations & Reports
    os.makedirs(output_reports_dir, exist_ok=True)
    os.makedirs(output_model_dir, exist_ok=True)

    # Chart 1: Model Comparison Chart
    comp_chart_path = os.path.join(output_reports_dir, "model_comparison.png")
    plot_model_comparison(comparison_df, output_path=comp_chart_path)

    # Chart 2: Confusion Matrix for Selected Model
    cm_chart_path = os.path.join(output_reports_dir, "confusion_matrix.png")
    selected_cm = eval_results[selected_model_name]["confusion_matrix"]["raw"]
    plot_confusion_matrix(selected_cm, selected_model_name, output_path=cm_chart_path)

    # Chart 3: Class Distribution Chart
    dist_chart_path = os.path.join(output_reports_dir, "class_distribution.png")
    plot_class_distribution(dataset_summary["class_distribution"], output_path=dist_chart_path)

    # Chart 4: ROC Curve for Selected Model
    roc_chart_path = os.path.join(output_reports_dir, "roc_curve.png")
    if hasattr(selected_model, "predict_proba"):
        selected_prob = selected_model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, selected_prob, selected_model_name, output_path=roc_chart_path)

    # 8. Save Evaluation Summary and Report
    full_eval_summary = {
        "training_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "selected_model_name": selected_model_name,
        "selection_criterion": "Highest F1-Score on held-out stratified test set",
        "dataset_summary": dataset_summary,
        "train_set_size": int(X_train.shape[0]),
        "test_set_size": int(X_test.shape[0]),
        "feature_names": FEATURE_NAMES,
        "models": eval_results,
        "comparison_table": comparison_df.to_dict(orient="records"),
    }
    save_evaluation_report(full_eval_summary, reports_dir=output_reports_dir)

    # 9. Persist Selected Model and Metadata
    model_save_path = os.path.join(output_model_dir, "phishing_detector_model.joblib")
    metadata_save_path = os.path.join(output_model_dir, "model_metadata.joblib")

    joblib.dump(selected_model, model_save_path)
    joblib.dump(full_eval_summary, metadata_save_path)
    logger.info(f"Saved trained model to {model_save_path}")
    logger.info(f"Saved metadata to {metadata_save_path}")

    logger.info("Pipeline execution successfully completed!")
    return full_eval_summary


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Phishing URL Detection models."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/mendeley_dataset.csv",
        help="Path to CSV dataset file (default: data/mendeley_dataset.csv)",
    )
    parser.add_argument(
        "--url-col",
        type=str,
        default=None,
        help="Explicit name of URL column (auto-detected if omitted)",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default=None,
        help="Explicit name of Label column (auto-detected if omitted)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25000,
        help="Max number of stratified samples to use for training (default: 25000; set 0 for all)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of test set split (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    max_samples = None if args.max_samples <= 0 else args.max_samples

    if not os.path.exists(args.dataset):
        logger.error(
            f"Dataset not found at {args.dataset}. "
            "Please ensure the dataset file exists in data/ directory."
        )
        sys.exit(1)

    train_and_evaluate_models(
        dataset_path=args.dataset,
        url_col=args.url_col,
        label_col=args.label_col,
        max_samples=max_samples,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
