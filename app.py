import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
import logging
import os

# ── Suppress all warnings and verbose output ──────────────────────────────────
warnings.filterwarnings("ignore")
logging.getLogger("shap").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)
os.environ["PYTHONWARNINGS"] = "ignore"

# ── Import from your own modules ─────────────────────────────────────────────
from data_preprocessing import preprocess_dataframe
from noise_detection import detect_anomalies
from shap_explainability import get_shap_sample, compute_shap_values
import shap

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Quality Management System",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Sidebar background ── */
    [data-testid="stSidebar"] { background-color: #2D4A6B !important; }
    [data-testid="stSidebar"] > div { background-color: #2D4A6B !important; }

    /* ── All sidebar text white ── */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] .stMarkdown {
        color: #F1F5F9 !important;
    }

    /* ── File uploader button in sidebar ── */
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: #3D5A7B !important;
        border-color: #5B7FA6 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background-color: #5B7FA6 !important;
        color: #FFFFFF !important;
        border: 1px solid #7B9FC6 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] p {
        color: #E2EBF6 !important;
    }

    /* ── Slider in sidebar ── */
    [data-testid="stSidebar"] .stSlider label { color: #F1F5F9 !important; }

    /* ── Expander in sidebar ── */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #F1F5F9 !important;
        background-color: #3D5A7B !important;
    }

    /* ── Download buttons ── */
    .stDownloadButton button {
        font-size: 1rem !important; font-weight: 700 !important;
        padding: 0.6rem 1.4rem !important; border-radius: 8px !important;
        background-color: #3B82F6 !important; color: white !important; border: none !important;
    }
    .stDownloadButton button:hover { background-color: #2563EB !important; }

    /* ── Divider ── */
    hr { margin: 2rem 0 !important; border-color: #CBD5E1 !important; }

    /* ── st.table dark text ── */
    table { width: 100%; border-collapse: collapse; }
    thead th {
        background-color: #1E3A5F !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        padding: 10px 14px !important;
        text-align: left !important;
    }
    tbody td {
        color: #0F172A !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 8px 14px !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }
    tbody tr:nth-child(even) { background-color: #F8FAFC !important; }
    tbody tr:hover { background-color: #EFF6FF !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-bottom: 1rem;">
    <h1 style="font-size:2.6rem; font-weight:800; color:#0F172A; margin-bottom:0.4rem; line-height:1.2;">
        🔍 Explainable AI-Based Data Quality Management System
    </h1>
    <p style="font-size:1.35rem; font-weight:600; color:#1E40AF; margin-bottom:0.6rem;">
        Automated Noise Detection in Numerical Data Pipelines
    </p>
    <p style="font-size:1.15rem; color:#040e1c; line-height:1.9; max-width:860px;">
        This system automatically detects noisy data points in numerical datasets using
        three machine learning algorithms and explains the results using
        SHAP (SHapley Additive exPlanations).
    </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("### 📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload your numerical CSV dataset",
    type=["csv"]
)

# ── Auto-suggest contamination rate using IQR method ─────────────────────────
def suggest_contamination(df):
    """
    Automatically estimate contamination rate using the IQR method.
    For each numerical column, count rows that fall outside 1.5*IQR range.
    The suggested rate = average outlier % across all numerical columns.
    Clamped between 0.01 and 0.20.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numerical_cols) == 0:
        return 0.05

    outlier_flags = pd.Series([False] * len(df))
    for col in numerical_cols:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_flags = outlier_flags | ((df[col] < lower) | (df[col] > upper))

    rate = round(outlier_flags.sum() / len(df), 2)
    return float(max(0.01, min(0.20, rate)))

# Compute suggestion if file is uploaded, else use default
if uploaded_file is not None:
    try:
        _df_preview = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)   # reset file pointer after preview read
        suggested_rate = suggest_contamination(_df_preview)
        st.sidebar.markdown("### 🎯 Contamination Rate")
        st.sidebar.caption(
            f"📌 **Auto-suggested: {int(suggested_rate*100)}%** based on IQR outlier analysis of your dataset. "
            f"You can adjust this if you have domain knowledge."
        )
    except:
        suggested_rate = 0.05
        st.sidebar.markdown("### 🎯 Contamination Rate")
else:
    suggested_rate = 0.05
    st.sidebar.markdown("### 🎯 Contamination Rate")
    st.sidebar.caption("Upload a dataset to get an auto-suggested contamination rate.")

contamination = st.sidebar.slider(
    "Adjust Contamination Rate",
    min_value=0.01, max_value=0.20,
    value=suggested_rate,
    step=0.01,
    help="This is the estimated % of noisy data points in your dataset. The value is auto-suggested based on IQR analysis but you can override it."
)

st.sidebar.caption(f"Current setting: **{int(contamination*100)}%** of data expected to be noisy.")

# ── Advanced Settings (hidden for beginners) ──────────────────────────────────
with st.sidebar.expander("⚙️ Advanced Settings"):
    shap_sample_size = st.slider(
        "SHAP Sample Size",
        min_value=100, max_value=1000, value=300, step=100,
        help="Number of rows used for SHAP explanation. Higher = more accurate but slower. Most users can leave this as default."
    )
    st.caption("Only change this if your dataset is very large and SHAP is taking too long.")

st.sidebar.divider()

# ── How to Use ────────────────────────────────────────────────────────────────
st.sidebar.markdown("### 📖 How to Use")
st.sidebar.markdown("""
<div style="font-size:1rem; line-height:2; color:#E2EBF6;">
1. 📂 &nbsp;<b>Upload</b> your numerical CSV file<br>
2. 🔢 &nbsp;<b>Select</b> the columns to analyse<br>
3. ⚙️ &nbsp;<b>Adjust</b> contamination rate if needed<br>
4. ✅ &nbsp;Results appear automatically!
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("### About")
st.sidebar.info("""
**Algorithms Used:**
- 🌲 Isolation Forest
- 🔍 Local Outlier Factor
- 🤖 One-Class SVM

**Explainability:**
- 📊 SHAP Summary Plot
- 🐝 SHAP Beeswarm Plot
- 🌊 SHAP Waterfall Plot
""")

# ── Main Pipeline ─────────────────────────────────────────────────────────────
if uploaded_file is not None:

    try:
        # ── Step 1: Data Overview ─────────────────────────────────────────────
        st.markdown('<h2 style="font-size:1.6rem;font-weight:700;color:#0F172A;border-left:5px solid #3B82F6;padding-left:12px;margin-top:2rem;margin-bottom:1rem;">📁 Step 1: Data Overview</h2>', unsafe_allow_html=True)
        df = pd.read_csv(uploaded_file)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows",     f"{df.shape[0]:,}")
        col2.metric("Total Columns",  f"{df.shape[1]}")
        col3.metric("Missing Values", f"{df.isnull().sum().sum()}")

        st.dataframe(df.head(10))

        # ── Step 2: Select Columns ────────────────────────────────────────────
        st.markdown('<h2 style="font-size:1.6rem;font-weight:700;color:#0F172A;border-left:5px solid #3B82F6;padding-left:12px;margin-top:2rem;margin-bottom:1rem;">🔢 Step 2: Select Numerical Columns</h2>', unsafe_allow_html=True)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) == 0:
            st.error("❌ No numerical columns found in your dataset!")
            st.stop()

        selected_cols = st.multiselect(
            "Select numerical columns for noise detection:",
            options=numerical_cols,
            default=numerical_cols[:5] if len(numerical_cols) >= 5 else numerical_cols
        )

        if len(selected_cols) == 0:
            st.warning("⚠️ Please select at least one column.")
            st.stop()

        # ── Step 3: Preprocessing ─────────────────────────────────────────────
        st.markdown('<h2 style="font-size:1.6rem;font-weight:700;color:#0F172A;border-left:5px solid #3B82F6;padding-left:12px;margin-top:2rem;margin-bottom:1rem;">🔧 Step 3: Preprocessing</h2>', unsafe_allow_html=True)

        with st.spinner("Preprocessing data..."):
            df_scaled, scaler, df_numerical = preprocess_dataframe(df, selected_cols)

        col1, col2 = st.columns(2)
        col1.success(f"✅ Rows after preprocessing: {len(df_numerical):,}")
        col2.success(f"✅ Columns selected: {len(selected_cols)}")

        st.dataframe(df_scaled.describe().round(4))

        # ── Step 4: Noise Detection ───────────────────────────────────────────
        st.markdown('<h2 style="font-size:1.6rem;font-weight:700;color:#0F172A;border-left:5px solid #3B82F6;padding-left:12px;margin-top:2rem;margin-bottom:1rem;">🚨 Step 4: Noise Detection</h2>', unsafe_allow_html=True)

        with st.spinner("🤖 Running all three models... this may take a moment..."):
            df_results, model_if, model_lof, model_svm = detect_anomalies(
                df_scaled, df_numerical, contamination
            )

        if_labels  = df_results['anomaly_if'].tolist()
        lof_labels = df_results['anomaly_lof'].tolist()
        svm_labels = df_results['anomaly_svm'].tolist()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌲 Isolation Forest", f"{sum(if_labels):,} anomalies",
                    f"{round(sum(if_labels)/len(if_labels)*100,2)}%")
        col2.metric("🔍 LOF", f"{sum(lof_labels):,} anomalies",
                    f"{round(sum(lof_labels)/len(lof_labels)*100,2)}%")
        col3.metric("🤖 One-Class SVM", f"{sum(svm_labels):,} anomalies",
                    f"{round(sum(svm_labels)/len(svm_labels)*100,2)}%")
        col4.metric("🤝 Consensus (2/3)", f"{df_results['anomaly_consensus'].sum():,} anomalies",
                    f"{round(df_results['anomaly_consensus'].mean()*100,2)}%")

        # ── Step 5: Data Quality Score + Outputs ─────────────────────────────
        st.markdown('<h2 style="font-size:1.6rem;font-weight:700;color:#0F172A;border-left:5px solid #3B82F6;padding-left:12px;margin-top:2rem;margin-bottom:1rem;">📊 Step 5: Data Quality Report & Downloads</h2>', unsafe_allow_html=True)

        # ── Compute quality score ─────────────────────────────────────────────
        total_rows  = len(df_numerical)
        total_cells = df_numerical.size

        anomaly_count      = df_results["anomaly_consensus"].sum()
        noise_score        = round((1 - anomaly_count / total_rows) * 100, 1)
        missing_cells      = df[selected_cols].isnull().sum().sum()
        completeness_score = round((1 - missing_cells / total_cells) * 100, 1)
        duplicate_count    = df[selected_cols].duplicated().sum()
        uniqueness_score   = round((1 - duplicate_count / total_rows) * 100, 1)
        overall_score      = round(
            (noise_score * 0.5 + completeness_score * 0.25 + uniqueness_score * 0.25), 1
        )

        if overall_score >= 90:
            score_color = "#16A34A"
            score_emoji = "🟢"
            score_label = "Excellent"
        elif overall_score >= 70:
            score_color = "#D97706"
            score_emoji = "🟡"
            score_label = "Moderate"
        else:
            score_color = "#DC2626"
            score_emoji = "🔴"
            score_label = "Poor"

        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid {score_color}; border-top:6px solid {score_color};
                    border-radius:14px; padding:2rem; text-align:center;
                    box-shadow:0 4px 12px rgba(0,0,0,0.08); margin-bottom:1.5rem;">
            <div style="font-size:1.1rem; font-weight:600; color:#475569; margin-bottom:0.3rem;">
                Overall Data Quality Score
            </div>
            <div style="font-size:4rem; font-weight:900; color:{score_color}; line-height:1.1;">
                {overall_score}%
            </div>
            <div style="font-size:1.2rem; color:{score_color}; font-weight:600; margin-bottom:1.5rem;">
                {score_emoji} {score_label}
            </div>
            <div style="display:flex; justify-content:center; gap:3rem; flex-wrap:wrap;">
                <div>
                    <div style="font-size:1.6rem; font-weight:800; color:#1E293B;">{noise_score}%</div>
                    <div style="font-size:0.95rem; color:#64748B; font-weight:500;">🚨 Noise-Free</div>
                </div>
                <div>
                    <div style="font-size:1.6rem; font-weight:800; color:#1E293B;">{completeness_score}%</div>
                    <div style="font-size:0.95rem; color:#64748B; font-weight:500;">✅ Completeness</div>
                </div>
                <div>
                    <div style="font-size:1.6rem; font-weight:800; color:#1E293B;">{uniqueness_score}%</div>
                    <div style="font-size:0.95rem; color:#64748B; font-weight:500;">🔁 Uniqueness</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Compute outputs ───────────────────────────────────────────────────
        with st.spinner("⏳ Computing SHAP reasons for all anomalies..."):
            anomaly_mask   = df_results["anomaly_consensus"] == 1
            normal_mask    = df_results["anomaly_consensus"] == 0

            df_scaled_reset    = df_scaled.reset_index(drop=True)
            df_numerical_reset = df_numerical.reset_index(drop=True)
            df_results_reset   = df_results.reset_index(drop=True)

            all_anomaly_idx         = df_results_reset[anomaly_mask].index.tolist()
            df_all_anomalies_scaled = df_scaled_reset.iloc[all_anomaly_idx].reset_index(drop=True)
            explainer_full          = shap.TreeExplainer(model_if)
            shap_vals_all           = explainer_full.shap_values(df_all_anomalies_scaled)
            feature_names           = selected_cols

            def build_reason(shap_row, feature_names):
                paired  = list(zip(feature_names, shap_row))
                ranked  = sorted(paired, key=lambda x: abs(x[1]), reverse=True)
                parts   = []
                for feat, val in ranked:
                    direction = "High" if val < 0 else "Low"
                    parts.append(f"{direction} {feat} (SHAP: {round(abs(val), 3)})")
                return " | ".join(parts)

            def build_flagged_by(row):
                models = []
                if row["anomaly_if"]  == 1: models.append("Isolation Forest")
                if row["anomaly_lof"] == 1: models.append("LOF")
                if row["anomaly_svm"] == 1: models.append("One-Class SVM")
                return ", ".join(models)

            df_anomaly_original = df_numerical_reset.iloc[all_anomaly_idx].reset_index(drop=True)
            df_anomaly_results  = df_results_reset.iloc[all_anomaly_idx].reset_index(drop=True)

            reasons    = [build_reason(shap_vals_all[i], feature_names) for i in range(len(all_anomaly_idx))]
            flagged_by = [build_flagged_by(df_anomaly_results.iloc[i]) for i in range(len(all_anomaly_idx))]

            df_anomaly_report = df_anomaly_original[selected_cols].copy()
            df_anomaly_report.insert(0, "flagged_by", flagged_by)
            df_anomaly_report["reason"] = reasons

            normal_idx = df_results_reset[normal_mask].index.tolist()
            df_cleaned = df_numerical_reset.iloc[normal_idx][selected_cols].reset_index(drop=True)

        # ── Output 1: Cleaned Dataset ─────────────────────────────────────────
        st.markdown('<h3 style="font-size:1.25rem;font-weight:600;color:#1E293B;margin-top:1.2rem;margin-bottom:0.5rem;">✅ Output 1: Cleaned Dataset (Noise Removed)</h3>', unsafe_allow_html=True)
        st.markdown(f"All **{len(df_cleaned):,}** normal data points after removing **{len(df_anomaly_report):,}** anomalies.")
        st.dataframe(df_cleaned.head(20))

        csv_cleaned = df_cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇️ Download Cleaned Dataset",
            data      = csv_cleaned,
            file_name = "cleaned_dataset.csv",
            mime      = "text/csv",
            key       = "download_cleaned"
        )

        st.divider()

        # ── Output 2: Anomaly Report ──────────────────────────────────────────
        st.markdown('<h3 style="font-size:1.25rem;font-weight:600;color:#1E293B;margin-top:1.2rem;margin-bottom:0.5rem;">🚨 Output 2: Anomaly Report (With Reasons)</h3>', unsafe_allow_html=True)
        st.markdown(f"**{len(df_anomaly_report):,}** anomalies detected. Each row shows which models flagged it and why, based on SHAP feature contributions.")
        st.dataframe(df_anomaly_report, width="stretch")

        csv_report = df_anomaly_report.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "⬇️ Download Anomaly Report",
            data      = csv_report,
            file_name = "anomaly_report.csv",
            mime      = "text/csv",
            key       = "download_report"
        )

        st.divider()
        st.success("✅ Core Pipeline Complete! Cleaned dataset and anomaly report are ready.")

        # ── Step 6: Detailed Technical Analysis (expandable) ─────────────────
        st.markdown("""
        <div style="background:#EFF6FF; border:1px solid #3B82F6; border-radius:10px;
                    padding:1rem 1.5rem; margin:1rem 0;">
            <p style="color:#1E40AF; font-size:1.05rem; font-weight:600; margin:0;">
                🔬 Want to explore the technical details? Expand the section below to see
                model comparisons, evaluation metrics, feature distributions and SHAP explanations.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔬 Detailed Technical Analysis", expanded=False):

            # ── Model Comparison ──────────────────────────────────────────────
            st.markdown('<h3 style="font-size:1.25rem;font-weight:600;color:#1E293B;margin-top:1.2rem;margin-bottom:0.5rem;">📊 Model Comparison</h3>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Anomaly Count Comparison**")
                fig, ax = plt.subplots(figsize=(7, 4))
                models = ['Isolation\nForest', 'LOF', 'One-Class\nSVM', 'Consensus']
                counts = [sum(if_labels), sum(lof_labels), sum(svm_labels),
                          df_results['anomaly_consensus'].sum()]
                colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
                bars = ax.bar(models, counts, color=colors, edgecolor='black')
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(counts) * 0.02,
                            f'{int(bar.get_height()):,}',
                            ha='center', fontsize=9, fontweight='bold')
                ax.set_ylabel('Anomalies Detected')
                ax.set_title('Model Comparison')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("**Model Agreement Heatmap**")
                fig, ax = plt.subplots(figsize=(7, 4))
                agreement_df = pd.DataFrame({
                    'IF':  if_labels,
                    'LOF': lof_labels,
                    'SVM': svm_labels
                })
                sns.heatmap(agreement_df.corr(), annot=True, fmt='.2f',
                            cmap='coolwarm', ax=ax, linewidths=0.5)
                ax.set_title('Model Agreement Heatmap')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ── Evaluation Metrics ────────────────────────────────────────────
            st.markdown('<h3 style="font-size:1.25rem;font-weight:600;color:#1E293B;margin-top:1.2rem;margin-bottom:0.5rem;">📐 Evaluation Metrics (Consensus as Ground Truth)</h3>', unsafe_allow_html=True)
            st.markdown("""
Each model is evaluated against the **consensus result** (2 out of 3 models agree) as the ground truth.
- **Precision** — out of all points this model flagged, how many does the consensus agree are anomalies?
- **Recall** — out of all consensus anomalies, how many did this model catch?
- **F1-Score** — overall performance score combining both (higher is better, max = 1.0)
""")
            from sklearn.metrics import precision_score, recall_score, f1_score

            consensus_labels = df_results['anomaly_consensus'].tolist()
            eval_data = []
            for model_name, preds in [
                ("Isolation Forest", if_labels),
                ("Local Outlier Factor", lof_labels),
                ("One-Class SVM", svm_labels)
            ]:
                p  = round(precision_score(consensus_labels, preds, zero_division=0), 3)
                r  = round(recall_score(consensus_labels, preds, zero_division=0), 3)
                f1 = round(f1_score(consensus_labels, preds, zero_division=0), 3)
                n  = sum(preds)
                eval_data.append({
                    "Model": model_name, "Anomalies Detected": n,
                    "Precision": p, "Recall": r, "F1-Score": f1
                })

            eval_df = pd.DataFrame(eval_data)
            st.dataframe(eval_df.set_index("Model"), width="stretch")

            best_model = eval_df.loc[eval_df["F1-Score"].idxmax(), "Model"]
            best_f1    = eval_df["F1-Score"].max()
            st.info(f"🏆 Best performing model: **{best_model}** with F1-Score of **{best_f1}**")

            st.divider()

            # ── Anomaly Distribution per Feature ──────────────────────────────
            st.markdown('<h3 style="font-size:1.25rem;font-weight:600;color:#1E293B;margin-top:1.2rem;margin-bottom:0.5rem;">📈 Anomaly Distribution per Feature</h3>', unsafe_allow_html=True)

            cols_per_row = 3
            rows = [selected_cols[i:i + cols_per_row]
                    for i in range(0, len(selected_cols), cols_per_row)]

            for row in rows:
                cols = st.columns(len(row))
                for col, feature in zip(cols, row):
                    with col:
                        fig, ax = plt.subplots(figsize=(5, 3))
                        normal  = df_results[df_results['anomaly_consensus'] == 0][feature]
                        anomaly = df_results[df_results['anomaly_consensus'] == 1][feature]
                        ax.hist(normal,  bins=40, alpha=0.6, color='#3498DB', label='Normal')
                        ax.hist(anomaly, bins=40, alpha=0.6, color='#E74C3C', label='Anomaly')
                        ax.set_title(f'{feature.upper()}', fontweight='bold')
                        ax.legend(fontsize=8)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

            st.divider()

            # ── SHAP Explainability ───────────────────────────────────────────
            st.markdown('<h3 style="font-size:1.25rem;font-weight:600;color:#1E293B;margin-top:1.2rem;margin-bottom:0.5rem;">🧠 SHAP Explainability</h3>', unsafe_allow_html=True)
            st.markdown("""
Explaining **why** specific data points were flagged as noisy across all three models.
- **Isolation Forest** uses TreeExplainer (fast, exact)
- **LOF and One-Class SVM** use KernelExplainer (model-agnostic approximation)
""")

            # Helper: draw waterfall for one anomaly
            def draw_waterfall(exp, df_sample, res_sample, anomaly_col):
                idx_list = res_sample[res_sample[anomaly_col] == 1].index.tolist()
                if len(idx_list) == 0:
                    st.warning("No anomalies found in sample for waterfall plot.")
                    return
                idx = idx_list[0]
                explanation = exp(df_sample.iloc[[idx]])
                fig, ax = plt.subplots(figsize=(10, 5))
                shap.waterfall_plot(
                    shap.Explanation(
                        values        = explanation.values[0],
                        base_values   = explanation.base_values[0],
                        data          = df_sample.iloc[idx].values,
                        feature_names = df_sample.columns.tolist()
                    ),
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Prepare shared sample
            with st.spinner("⏳ Preparing SHAP sample..."):
                df_shap_sample, res_shap_sample = get_shap_sample(
                    df_scaled, df_results, shap_sample_size
                )
                bg_size = min(50, len(df_shap_sample))
                df_bg   = df_shap_sample.sample(n=bg_size, random_state=42)

            # 7A — Isolation Forest
            st.markdown("#### 🌲 Isolation Forest — SHAP Explanation")
            st.caption("Uses TreeExplainer — fast and exact.")

            with st.spinner("Computing Isolation Forest SHAP values..."):
                explainer_if, shap_values_if = compute_shap_values(model_if, df_shap_sample)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**SHAP Summary Plot (Bar)**")
                fig, ax = plt.subplots(figsize=(7, 5))
                shap.summary_plot(shap_values_if, df_shap_sample, plot_type="bar", show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            with col2:
                st.markdown("**SHAP Beeswarm Plot**")
                fig, ax = plt.subplots(figsize=(7, 5))
                shap.summary_plot(shap_values_if, df_shap_sample, plot_type="dot", show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.markdown("**SHAP Waterfall Plot — Single Anomaly**")
            st.caption("Why was this specific point flagged by Isolation Forest?")
            draw_waterfall(explainer_if, df_shap_sample, res_shap_sample, "anomaly_if")

            st.divider()

            # 7B — LOF
            st.markdown("#### 🔍 Local Outlier Factor — SHAP Explanation")
            st.caption("Uses KernelExplainer — model-agnostic approximation. May take a moment.")

            with st.spinner("Computing LOF SHAP values..."):
                try:
                    def lof_predict(X):
                        import pandas as pd
                        X_df = pd.DataFrame(X, columns=df_shap_sample.columns)
                        dists, _ = model_lof.kneighbors(X_df)
                        return -dists.mean(axis=1)

                    explainer_lof   = shap.KernelExplainer(lof_predict, df_bg)
                    shap_values_lof = explainer_lof.shap_values(df_shap_sample, nsamples=100)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**SHAP Summary Plot (Bar)**")
                        fig, ax = plt.subplots(figsize=(7, 5))
                        shap.summary_plot(shap_values_lof, df_shap_sample, plot_type="bar", show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    with col2:
                        st.markdown("**SHAP Beeswarm Plot**")
                        fig, ax = plt.subplots(figsize=(7, 5))
                        shap.summary_plot(shap_values_lof, df_shap_sample, plot_type="dot", show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    st.markdown("**SHAP Waterfall Plot — Single Anomaly**")
                    st.caption("Why was this specific point flagged by LOF?")
                    lof_anomaly_idx = res_shap_sample[res_shap_sample["anomaly_lof"] == 1].index.tolist()
                    if len(lof_anomaly_idx) > 0:
                        idx = lof_anomaly_idx[0]
                        fig, ax = plt.subplots(figsize=(10, 5))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values        = shap_values_lof[idx],
                                base_values   = explainer_lof.expected_value,
                                data          = df_shap_sample.iloc[idx].values,
                                feature_names = df_shap_sample.columns.tolist()
                            ),
                            show=False
                        )
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.warning("No LOF anomalies found in sample.")
                except Exception as lof_err:
                    st.warning(f"LOF SHAP could not be computed: {lof_err}")

            st.divider()

            # 7C — One-Class SVM
            st.markdown("#### 🤖 One-Class SVM — SHAP Explanation")
            st.caption("Uses KernelExplainer — model-agnostic approximation. May take a moment.")

            with st.spinner("Computing One-Class SVM SHAP values..."):
                try:
                    def svm_predict(X):
                        import pandas as pd
                        X_df = pd.DataFrame(X, columns=df_shap_sample.columns)
                        return model_svm.decision_function(X_df)

                    explainer_svm   = shap.KernelExplainer(svm_predict, df_bg)
                    shap_values_svm = explainer_svm.shap_values(df_shap_sample, nsamples=100)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**SHAP Summary Plot (Bar)**")
                        fig, ax = plt.subplots(figsize=(7, 5))
                        shap.summary_plot(shap_values_svm, df_shap_sample, plot_type="bar", show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    with col2:
                        st.markdown("**SHAP Beeswarm Plot**")
                        fig, ax = plt.subplots(figsize=(7, 5))
                        shap.summary_plot(shap_values_svm, df_shap_sample, plot_type="dot", show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    st.markdown("**SHAP Waterfall Plot — Single Anomaly**")
                    st.caption("Why was this specific point flagged by One-Class SVM?")
                    svm_anomaly_idx = res_shap_sample[res_shap_sample["anomaly_svm"] == 1].index.tolist()
                    if len(svm_anomaly_idx) > 0:
                        idx = svm_anomaly_idx[0]
                        fig, ax = plt.subplots(figsize=(10, 5))
                        shap.waterfall_plot(
                            shap.Explanation(
                                values        = shap_values_svm[idx],
                                base_values   = explainer_svm.expected_value,
                                data          = df_shap_sample.iloc[idx].values,
                                feature_names = df_shap_sample.columns.tolist()
                            ),
                            show=False
                        )
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.warning("No SVM anomalies found in sample.")
                except Exception as svm_err:
                    st.warning(f"SVM SHAP could not be computed: {svm_err}")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        st.info("Please check your dataset and try again. Make sure the file has numerical columns.")



else:
    st.markdown("""
    <div style="background:linear-gradient(135deg, #DBEAFE 0%, #EFF6FF 100%);
                border:1px solid #3B82F6; border-radius:16px;
                padding:2.5rem 2rem; text-align:center; margin-bottom:2rem;">
        <h2 style="color:#1E3A8A; font-size:1.7rem; font-weight:800; margin-bottom:0.5rem;">
            👈 Upload your CSV dataset from the sidebar to begin
        </h2>
        <p style="color:#3B82F6; font-size:1.1rem; margin:0; font-weight:500;">
            The system will automatically detect noisy data points and explain why each was flagged.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:#0F172A; font-size:1.5rem; font-weight:700; margin-bottom:1rem;'>⚡ How it works</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    cards = [
        ("1️⃣", "Upload",  "#3B82F6", "Upload any numerical CSV dataset from the sidebar"),
        ("2️⃣", "Detect",  "#10B981", "Three ML models automatically detect noisy data points"),
        ("3️⃣", "Compare", "#F59E0B", "Compare results across all three models visually"),
        ("4️⃣", "Explain", "#8B5CF6", "SHAP explains exactly why each point was flagged"),
    ]

    for col, (icon, title, color, desc) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""
            <div style="background-color:#FFFFFF; border:1px solid {color}; border-top:5px solid {color};
                        border-radius:12px; padding:1.8rem 1.2rem; min-height:200px; text-align:center;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="font-size:2.2rem; margin-bottom:0.6rem;">{icon}</div>
                <div style="color:{color}; font-size:1.2rem; font-weight:700; margin-bottom:0.8rem;">{title}</div>
                <div style="color:#475569; font-size:1rem; line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)