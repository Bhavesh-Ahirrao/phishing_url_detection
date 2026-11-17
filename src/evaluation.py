"""
Model Evaluation and Visualization Module.

Computes classification metrics, classification reports, confusion matrices,
and generates charts saved to the reports/ directory.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend safe for servers/scripts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Compute accuracy, precision, recall, F1, and ROC-AUC.
    Phishing (1) is treated as the positive class.
    
    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Optional predicted probabilities for positive class (1).
        
    Returns:
        Dictionary of computed metric values.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    auc = None
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=["Legitimate (0)", "Phishing (1)"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=["Legitimate (0)", "Phishing (1)"],
        zero_division=0,
    )

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "raw": cm.tolist(),
        },
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
    }


def plot_model_comparison(
    comparison_df: pd.DataFrame, output_path: str = "reports/model_comparison.png"
) -> str:
    """
    Plot grouped bar chart comparing Accuracy, Precision, Recall, and F1 across models.
    
    Args:
        comparison_df: DataFrame with columns ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        output_path: File path to save the chart.
        
    Returns:
        The output path string.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    melted = pd.melt(
        comparison_df,
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score",
    )

    plt.figure(figsize=(10, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    ax = sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette=palette)
    
    plt.title("Model Performance Comparison (Test Set)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Classification Algorithm", fontsize=11, fontweight="bold")
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="bold")
    plt.ylim(0, 1.08)
    plt.legend(loc="lower right", frameon=True)
    
    # Annotate bar values
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Model comparison chart saved to {output_path}")
    return output_path


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: str = "reports/confusion_matrix.png",
) -> str:
    """
    Plot and save confusion matrix heatmap with True Positives, False Positives, etc.
    
    Args:
        cm: 2x2 confusion matrix array.
        model_name: Name of the model.
        output_path: Path to save the figure.
        
    Returns:
        The output path string.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    labels = ["Legitimate (0)", "Phishing (1)"]
    group_names = ["True Neg (TN)", "False Pos (FP)", "False Neg (FN)", "True Pos (TP)"]
    
    cm_arr = np.array(cm)
    group_counts = [f"{value:0,d}" for value in cm_arr.flatten()]
    group_percentages = [f"{value:.2%}" for value in cm_arr.flatten() / np.sum(cm_arr)]
    
    box_labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(2, 2)

    plt.figure(figsize=(7, 6), dpi=300)
    sns.set_theme(style="white")
    
    sns.heatmap(
        cm_arr,
        annot=box_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 11, "weight": "bold"},
    )
    
    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Actual True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")
    return output_path


def plot_class_distribution(
    class_counts: Dict[str, int], output_path: str = "reports/class_distribution.png"
) -> str:
    """
    Plot and save class distribution bar chart.
    
    Args:
        class_counts: Dictionary mapping class name to count.
        output_path: Output path.
        
    Returns:
        The output path string.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    labels = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(7, 5), dpi=300)
    sns.set_theme(style="whitegrid")
    
    colors = ["#2b5c8f", "#d9534f"]
    bars = plt.bar(labels, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    
    plt.title("Dataset Class Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Class Category", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Samples", fontsize=11, fontweight="bold")
    
    max_c = max(counts) if counts else 100
    plt.ylim(0, max_c * 1.15)
    
    for bar in bars:
        height = bar.get_height()
        pct = (height / sum(counts)) * 100 if sum(counts) > 0 else 0
        plt.annotate(
            f"{int(height):,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Class distribution chart saved to {output_path}")
    return output_path


def plot_roc_curve(
    y_true, y_prob, model_name: str, output_path: str = "reports/roc_curve.png"
) -> str:
    """
    Plot and save Receiver Operating Characteristic (ROC) curve.
    
    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probability for positive class (1).
        model_name: Model name.
        output_path: Output path.
        
    Returns:
        The output path string.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(7, 6), dpi=300)
    sns.set_theme(style="whitegrid")
    
    plt.plot(fpr, tpr, color="#0077b6", lw=2.5, label=f"{model_name} (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="#888888", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title(f"ROC Curve: {model_name}", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")
    return output_path


def save_evaluation_report(
    eval_summary: Dict[str, Any],
    reports_dir: str = "reports",
) -> None:
    """
    Save complete evaluation summary to JSON and readable TXT format.
    
    Args:
        eval_summary: Full evaluation summary dictionary.
        reports_dir: Reports folder path.
    """
    os.makedirs(reports_dir, exist_ok=True)
    
    json_path = os.path.join(reports_dir, "evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=4)
    logger.info(f"Saved evaluation JSON to {json_path}")

    txt_path = os.path.join(reports_dir, "evaluation_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("INTELLIGENT PHISHING URL DETECTION SYSTEM - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Selected Best Model: {eval_summary.get('selected_model_name')}\n")
        f.write(f"Selection Criterion: {eval_summary.get('selection_criterion')}\n")
        f.write(f"Dataset Records: {eval_summary.get('dataset_summary', {}).get('final_records')}\n")
        f.write(f"Test Split Size: {eval_summary.get('test_set_size')}\n\n")

        f.write("-" * 70 + "\n")
        f.write("1. MODEL COMPARISON (TEST SET)\n")
        f.write("-" * 70 + "\n")
        
        models_metrics = eval_summary.get("models", {})
        header = f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}\n"
        f.write(header)
        f.write("-" * 75 + "\n")
        for m_name, m_data in models_metrics.items():
            auc_str = f"{m_data.get('roc_auc', 'N/A')}" if m_data.get('roc_auc') is not None else "N/A"
            f.write(
                f"{m_name:<25} "
                f"{m_data.get('accuracy', 0):<10.4f} "
                f"{m_data.get('precision', 0):<10.4f} "
                f"{m_data.get('recall', 0):<10.4f} "
                f"{m_data.get('f1_score', 0):<10.4f} "
                f"{auc_str:<10}\n"
            )
        f.write("\n")

        f.write("-" * 70 + "\n")
        f.write(f"2. CLASSIFICATION REPORT FOR SELECTED MODEL ({eval_summary.get('selected_model_name')})\n")
        f.write("-" * 70 + "\n")
        best_name = eval_summary.get("selected_model_name")
        if best_name and best_name in models_metrics:
            f.write(models_metrics[best_name].get("classification_report_text", "N/A"))
            cm_info = models_metrics[best_name].get("confusion_matrix", {})
            f.write("\n\nConfusion Matrix Breakdown:\n")
            f.write(f"  True Negatives (TN - Correct Legitimate):  {cm_info.get('tn')}\n")
            f.write(f"  False Positives (FP - False Alarm):       {cm_info.get('fp')}\n")
            f.write(f"  False Negatives (FN - Missed Phishing):   {cm_info.get('fn')}\n")
            f.write(f"  True Positives (TP - Caught Phishing):     {cm_info.get('tp')}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("SECURITY METRIC DISCUSSION: RECALL VS PRECISION TRADE-OFF\n")
        f.write("=" * 70 + "\n")
        f.write(
            "In phishing detection, Recall is of critical importance because a False Negative\n"
            "(failing to flag a phishing URL) exposes users directly to credential theft, fraud,\n"
            "or malware infection. Conversely, a False Positive disrupts user workflow with a false alarm.\n"
            "Therefore, models are evaluated primarily on F1-Score to balance both, ensuring high\n"
            "threat detection while maintaining acceptable precision.\n"
        )
    logger.info(f"Saved human-readable evaluation report to {txt_path}")
