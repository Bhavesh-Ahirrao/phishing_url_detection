"""
Intelligent Phishing URL Detection System - Streamlit Web Dashboard.

Provides an interactive cybersecurity dashboard for:
- Home & Project Architecture Overview
- Real-Time URL Threat Analysis & Feature Inspection
- Model Evaluation & Performance Analytics
- Engineering & Academic Project Documentation
"""

from __future__ import annotations

import os
import json
import pandas as pd
import streamlit as st
from PIL import Image

from src.prediction import URLPredictor
from src.feature_extraction import FEATURE_NAMES

# Page configuration
st.set_page_config(
    page_title="PhishShield ML | Phishing URL Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Cyber Dark Theme CSS
CUSTOM_CSS = """
<style>
    /* Main Background and Typography */
    .stApp {
        background: linear-gradient(135deg, #070d1e 0%, #0c1836 50%, #0a1329 100%);
        color: #e0e6ed;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #081024 !important;
        border-right: 1px solid #1a2a4f;
    }
    
    /* Headers & Accents */
    h1, h2, h3, h4, h5, h6 {
        color: #e6f1ff !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .glow-header {
        color: #48cae4 !important;
        text-shadow: 0 0 15px rgba(72, 202, 228, 0.4);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(14, 26, 56, 0.7);
        border: 1px solid #1d3557;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00b4d8;
        transform: translateY(-2px);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #90e0ef;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
    }
    
    /* Result Status Banners */
    .result-banner {
        padding: 24px;
        border-radius: 12px;
        margin: 20px 0;
        border: 1px solid;
    }
    .banner-phishing {
        background: rgba(220, 53, 69, 0.15);
        border-color: #e63946;
        color: #ffb4a2;
    }
    .banner-legitimate {
        background: rgba(40, 167, 69, 0.15);
        border-color: #2a9d8f;
        color: #b7e4c7;
    }
    .banner-warning {
        background: rgba(255, 193, 7, 0.15);
        border-color: #ffb703;
        color: #ffe6a7;
    }
    
    /* Notice and Disclaimer Box */
    .disclaimer-box {
        background: rgba(13, 27, 62, 0.6);
        border-left: 4px solid #ffb703;
        padding: 12px 18px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #ced4da;
        margin-top: 15px;
    }
    
    /* Tables */
    div[data-testid="stTable"] table {
        background-color: #0d1b3e !important;
        color: #e0e6ed !important;
        border-radius: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Load and cache the trained URL Predictor instance."""
    return URLPredictor(model_dir="models")


def load_evaluation_data():
    """Load cached evaluation summary and metrics JSON."""
    summary_path = os.path.join("reports", "evaluation_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def render_sidebar():
    """Render application sidebar with brand and navigation."""
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 10px 0 20px 0;">
                <div style="font-size: 40px; line-height: 1;">🛡️</div>
                <h2 style="margin: 6px 0 0 0; color: #48cae4; font-size: 1.4rem;">PhishShield ML</h2>
                <div style="font-size: 0.8rem; color: #90e0ef;">Intelligent URL Threat Classifier</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Navigation")
        page = st.radio(
            label="Go to Page",
            options=["Home Dashboard", "URL Threat Detection", "Model Performance", "About Project"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        predictor = get_predictor()
        if predictor.is_model_loaded:
            model_name = predictor.metadata.get("selected_model_name", "Classifier")
            st.success(f"Active Model: **{model_name}**")
        else:
            st.error("No active model loaded.")
            st.caption("Run `python train_model.py` in terminal.")

        st.markdown("---")
        st.caption("AI & Data Science Engineering Project\nSupervised Lexical URL Detection Pipeline")
        return page


# ==========================================
# PAGE 1: HOME DASHBOARD
# ==========================================
def render_home_page():
    st.markdown('<h1 class="glow-header">🛡️ Intelligent Phishing URL Detection System</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        An end-to-end, machine-learning-driven threat intelligence system designed to detect deceptive 
        and malicious phishing URLs in real time by analyzing their lexical, structural, and syntactic properties.
        """
    )

    eval_data = load_evaluation_data()
    predictor = get_predictor()

    # Metrics Summary Row
    st.markdown("### 📊 Operational Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        model_name = eval_data.get("selected_model_name", "Gradient Boosting") if eval_data else "Not Trained"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Selected Classifier</div>
                <div class="metric-val" style="font-size: 1.3rem; color: #00b4d8;">{model_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        acc = f"{eval_data['models'][model_name]['accuracy']*100:.2f}%" if eval_data and model_name in eval_data.get('models', {}) else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Test Accuracy</div>
                <div class="metric-val" style="color: #2a9d8f;">{acc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        f1 = f"{eval_data['models'][model_name]['f1_score']:.4f}" if eval_data and model_name in eval_data.get('models', {}) else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Test F1-Score</div>
                <div class="metric-val" style="color: #ffb703;">{f1}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        records = f"{eval_data['dataset_summary']['final_records']:,}" if eval_data and 'dataset_summary' in eval_data else "100k+ Dataset"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Evaluated Records</div>
                <div class="metric-val" style="font-size: 1.5rem; color: #e76f51;">{records}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Threat Architecture & Workflow
    col_arch1, col_arch2 = st.columns([1.2, 1])

    with col_arch1:
        st.markdown("### 🎯 What is Phishing?")
        st.write(
            """
            Phishing is a deceptive cybercrime where attackers craft fraudulent websites disguised 
            as legitimate services (such as banking institutions, email providers, and social platforms) 
            to trick victims into disclosing sensitive credentials, financial details, or personal data.
            """
        )
        st.markdown("### ⚙️ How the ML Pipeline Works")
        st.markdown(
            """
            1. **Safe Input Ingestion:** The input URL is stripped, validated, and normalized without making network connections.
            2. **Lexical & Structural Feature Extraction:** 22 numerical features are derived (domain length, delimiter frequencies, entropy indicators, suspicious keywords, and IP host checks).
            3. **Vector Alignment:** Identical feature schema used during training is reconstructed.
            4. **Ensemble Inference:** Pre-trained Supervised ML Model classifies the vector into Legitimate or Phishing with confidence estimation.
            """
        )

    with col_arch2:
        st.markdown("### 🔄 End-to-End System Workflow")
        workflow_steps = [
            ("1. Input URL", "User enters arbitrary URL string"),
            ("2. Safe Parsing", "Offline urllib & tldextract decomposition"),
            ("3. Feature Extraction", "22 syntactic, lexical & structural metrics"),
            ("4. ML Prediction", "Gradient Boosting / Random Forest Inference"),
            ("5. Threat Assessment", "Risk level, probability score & advisory"),
        ]
        for step, desc in workflow_steps:
            st.markdown(
                f"""
                <div style="background: rgba(13, 27, 62, 0.5); border-left: 3px solid #00b4d8; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                    <strong style="color: #90e0ef;">{step}:</strong> <span style="color: #cbd5e1;">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### ⚠️ Security Notice & Limitation Disclaimer")
    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>Statistical Estimate Disclaimer:</strong> This application relies exclusively on static URL 
            structural features. It does NOT establish live communication with the web host, inspect HTTP payloads, 
            or verify real-time domain registration status. The presence of <code>https://</code> only denotes 
            transport encryption, not website legitimacy. Never consider a machine learning score as an infallible 
            security warranty.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# PAGE 2: REAL-TIME URL DETECTION
# ==========================================
def render_detection_page():
    st.markdown('<h1 class="glow-header">🔍 Real-Time URL Threat Analysis</h1>', unsafe_allow_html=True)
    st.write(
        "Enter any raw URL to analyze its syntactic and lexical structure for phishing characteristics. "
        "The URL will be safely decomposed offline without opening or visiting the destination."
    )

    predictor = get_predictor()

    if not predictor.is_model_loaded:
        st.error(
            "Trained model not found in `models/`. Please train the model first by executing: "
            "`python train_model.py`"
        )
        return

    # Quick test samples
    st.markdown("##### 🧪 Quick Test Presets:")
    col_preset1, col_preset2, col_preset3 = st.columns(3)
    preset_url = ""
    with col_preset1:
        if st.button("Example 1: Legitimate Domain (Wikipedia)", use_container_width=True):
            preset_url = "https://en.wikipedia.org/wiki/Phishing"
    with col_preset2:
        if st.button("Example 2: Credential Phishing Lure", use_container_width=True):
            preset_url = "http://login-verify-account-update.servehttp.com/banking/signin.php?token=9281"
    with col_preset3:
        if st.button("Example 3: IP Address & Suspicious Port", use_container_width=True):
            preset_url = "http://192.168.1.100:8888/account-security/verification"

    url_input = st.text_input(
        "Enter URL to inspect:",
        value=preset_url,
        placeholder="e.g., https://secure-login-portal.example.com/account/update",
        help="Paste a complete or schemeless URL. The system will evaluate it safely.",
    )

    analyze_clicked = st.button("🚀 Analyze URL", type="primary", use_container_width=True)

    if analyze_clicked or (preset_url and not url_input):
        eval_url = url_input or preset_url
        if not eval_url.strip():
            st.warning("Please enter a valid URL to analyze.")
            return

        with st.spinner("Decomposing URL and calculating structural features..."):
            result = predictor.predict(eval_url)

        status = result["status"]
        prob = result["phishing_probability"]
        risk = result["risk_level"]
        features = result["features"]

        # Classification Status Render
        if status == "Potentially Phishing":
            st.markdown(
                f"""
                <div class="result-banner banner-phishing">
                    <h2 style="color: #ff4d6d; margin: 0 0 10px 0;">🚨 Alert: Potentially Phishing URL Detected</h2>
                    <p style="font-size: 1.05rem; margin: 0 0 8px 0;">{result['message']}</p>
                    <div><strong>Risk Level:</strong> <span style="background: #e63946; color: white; padding: 3px 10px; border-radius: 4px;">{risk}</span>
                    &nbsp;&nbsp;<strong>Phishing Probability:</strong> {prob * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif status == "Legitimate":
            st.markdown(
                f"""
                <div class="result-banner banner-legitimate">
                    <h2 style="color: #2ec4b6; margin: 0 0 10px 0;">✅ Result: Likely Legitimate URL</h2>
                    <p style="font-size: 1.05rem; margin: 0 0 8px 0;">{result['message']}</p>
                    <div><strong>Risk Level:</strong> <span style="background: #2a9d8f; color: white; padding: 3px 10px; border-radius: 4px;">{risk}</span>
                    &nbsp;&nbsp;<strong>Safety Confidence:</strong> {(1 - prob) * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="result-banner banner-warning">
                    <h2 style="color: #ffb703; margin: 0 0 10px 0;">⚠️ {status}</h2>
                    <p style="margin: 0;">{result['message']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if prob is not None:
            st.markdown("#### Probability & Confidence Gauge")
            st.progress(prob, text=f"Estimated Phishing Probability: {prob * 100:.1f}%")

        # Feature Breakdown Expander
        if features:
            with st.expander("🔍 View Extracted URL Features (22 Structural Indicators)", expanded=False):
                st.write("Below are the exact numerical feature values extracted by the URL parser:")
                
                # Split features into 2 columns for neat viewing
                feat_items = list(features.items())
                half = len(feat_items) // 2
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    f_df1 = pd.DataFrame(feat_items[:half], columns=["Feature Name", "Extracted Value"])
                    st.dataframe(f_df1, use_container_width=True, hide_index=True)
                with col_f2:
                    f_df2 = pd.DataFrame(feat_items[half:], columns=["Feature Name", "Extracted Value"])
                    st.dataframe(f_df2, use_container_width=True, hide_index=True)

        st.markdown(
            f"""
            <div class="disclaimer-box">
                {result['disclaimer']}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==========================================
# PAGE 3: MODEL PERFORMANCE
# ==========================================
def render_performance_page():
    st.markdown('<h1 class="glow-header">📈 Model Performance & Comparative Evaluation</h1>', unsafe_allow_html=True)
    st.write(
        "Comprehensive empirical evaluation across supervised machine learning algorithms "
        "trained and validated on held-out stratified test data."
    )

    eval_data = load_evaluation_data()

    if not eval_data:
        st.warning(
            "Evaluation report not found in `reports/evaluation_summary.json`. "
            "Please run `python train_model.py` to generate complete evaluation metrics."
        )
        return

    # Algorithm Comparison Table
    st.markdown("### 🏆 Algorithm Comparison (Held-Out Test Set)")
    comparison_table = eval_data.get("comparison_table", [])
    if comparison_table:
        comp_df = pd.DataFrame(comparison_table)
        st.dataframe(
            comp_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
            }).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#1d3557"),
            use_container_width=True,
            hide_index=True,
        )

    selected_model_name = eval_data.get("selected_model_name", "Selected Model")
    st.info(
        f"**Selected Model:** `{selected_model_name}` chosen based on: "
        f"*{eval_data.get('selection_criterion', 'Highest F1-Score')}*."
    )

    # Visualizations
    st.markdown("---")
    st.markdown("### 📊 Evaluation Charts & Visual Diagnostics")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        comp_img_path = os.path.join("reports", "model_comparison.png")
        if os.path.exists(comp_img_path):
            st.image(comp_img_path, caption="Algorithm Performance Comparison (Test Set)", use_column_width=True)

    with col_chart2:
        cm_img_path = os.path.join("reports", "confusion_matrix.png")
        if os.path.exists(cm_img_path):
            st.image(cm_img_path, caption=f"Confusion Matrix: {selected_model_name}", use_column_width=True)

    col_chart3, col_chart4 = st.columns(2)
    with col_chart3:
        roc_img_path = os.path.join("reports", "roc_curve.png")
        if os.path.exists(roc_img_path):
            st.image(roc_img_path, caption=f"ROC Curve & AUC: {selected_model_name}", use_column_width=True)

    with col_chart4:
        dist_img_path = os.path.join("reports", "class_distribution.png")
        if os.path.exists(dist_img_path):
            st.image(dist_img_path, caption="Dataset Class Distribution", use_column_width=True)

    # Classification Report
    st.markdown("---")
    st.markdown(f"### 📋 Detailed Classification Report: `{selected_model_name}`")
    best_model_data = eval_data.get("models", {}).get(selected_model_name, {})
    if "classification_report_text" in best_model_data:
        st.code(best_model_data["classification_report_text"], language="text")

    # Metric Discussion
    st.markdown("### 💡 Why Recall Matters in Cybersecurity Detection")
    st.markdown(
        """
        - **False Negative (FN):** Phishing URL classified as Legitimate. The victim accesses the malicious site, leading to credential theft, malware infection, or corporate compromise.
        - **False Positive (FP):** Legitimate URL flagged as Phishing. Creates user inconvenience or false alarm alert fatigue.
        - **F1-Score Balance:** In security operations, optimizing purely for Accuracy hides dangerous false negative blindspots. F1-score provides the harmonized balance between catching all threats (Recall) and minimizing false alarms (Precision).
        """
    )


# ==========================================
# PAGE 4: ABOUT THE PROJECT
# ==========================================
def render_about_page():
    st.markdown('<h1 class="glow-header">📖 About the Intelligent Phishing Detection Project</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        ### 1. Problem Statement
        Phishing remains the predominant initial attack vector in cyber warfare and cybercrime. 
        Traditional blacklist-based countermeasures (e.g., DNS blocklists) fail against zero-day phishing 
        domains generated rapidly using dynamic DNS, domain generation algorithms (DGAs), and automated cloud hosting. 
        This project addresses this vulnerability by developing a machine-learning-based classification pipeline 
        that evaluates structural and lexical URL characteristics.
        """
    )

    st.markdown(
        """
        ### 2. Project Objectives
        - Build an end-to-end reproducible Machine Learning system for automated phishing URL identification.
        - Extract 22 domain, delimiter, structural, and keyword indicators without network exposure.
        - Train and benchmark four distinct supervised classification algorithms.
        - Persist model artifacts safely with Joblib for sub-millisecond inference.
        - Deliver a responsive Streamlit demonstration dashboard suitable for academic engineering presentation.
        """
    )

    st.markdown(
        """
        ### 3. Machine Learning Algorithms Implemented
        1. **Logistic Regression:** Linear probabilistic baseline with standardized input features.
        2. **Decision Tree Classifier:** Non-linear rule-based classifier uncovering hierarchical decision thresholds.
        3. **Random Forest Classifier:** Bagging ensemble combining 120 randomized decision trees to reduce variance.
        4. **HistGradientBoosting Classifier:** High-efficiency gradient-boosted decision tree algorithm optimizing pseudo-residuals.
        """
    )

    st.markdown(
        """
        ### 4. Feature Engineering Process
        Features are categorized into:
        - **Lexical Dimensions:** Overall URL length, hostname length, registered domain length.
        - **Syntactic Delimiters:** Counts of dots (`.`), hyphens (`-`), underscores (`_`), slashes (`/`), question marks (`?`), equals (`=`), ampersands (`&`), at-symbols (`@`), percent symbols (`%`).
        - **Structural Indicators:** Subdomain depth, IP address as hostname, non-standard port designation.
        - **Semantic Tokens:** Presence of credential-harvesting lures (`login`, `verify`, `update`, `banking`, `secure`, `wallet`).
        """
    )

    st.markdown(
        """
        ### 5. Academic Project Limitations & Future Scope
        - **Homograph Attacks:** Advanced Unicode punycode homoglyphs require dedicated normalization.
        - **Shortened URLs:** Services like `bit.ly` or `t.co` conceal structural characteristics without HTTP unshortening.
        - **Future Scope:** Integration of explainable AI (SHAP/LIME), real-time DNS reputation feeds, and deep contextual sequence models (Transformers/BERT for URLs).
        """
    )


# Main Entry Point Router
def main():
    selected_page = render_sidebar()

    if selected_page == "Home Dashboard":
        render_home_page()
    elif selected_page == "URL Threat Detection":
        render_detection_page()
    elif selected_page == "Model Performance":
        render_performance_page()
    elif selected_page == "About Project":
        render_about_page()


if __name__ == "__main__":
    main()
