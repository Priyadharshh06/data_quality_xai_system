import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import random
import os
from noise_detection import run_noise_detection

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SHAP_SAMPLE_SIZE = 500


def get_shap_sample(df_scaled, df_results, shap_sample_size=500):
    """
    Get a balanced sample of normal and anomaly points for SHAP analysis.
    Reusable — accepts any df_scaled and df_results.
    """
    df_scaled  = df_scaled.reset_index(drop=True)
    df_results = df_results.reset_index(drop=True)

    anomaly_idx = df_results[df_results['anomaly_if'] == 1].index.tolist()
    normal_idx  = df_results[df_results['anomaly_if'] == 0].index.tolist()

    n_each = min(shap_sample_size // 2, len(anomaly_idx), len(normal_idx))

    random.seed(42)
    sampled_idx = random.sample(anomaly_idx, n_each) + random.sample(normal_idx, n_each)

    df_sample  = df_scaled.iloc[sampled_idx].reset_index(drop=True)
    res_sample = df_results.iloc[sampled_idx].reset_index(drop=True)

    print(f" SHAP sample ready: {len(df_sample)} rows ({n_each} anomalies + {n_each} normal)")
    return df_sample, res_sample


def compute_shap_values(model_if, df_sample):
    """Compute SHAP values using TreeExplainer on Isolation Forest."""
    print(" Computing SHAP values...")
    explainer   = shap.TreeExplainer(model_if)
    shap_values = explainer.shap_values(df_sample)
    print(" SHAP values computed!")
    return explainer, shap_values


def plot_shap_summary(shap_values, df_sample):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, df_sample, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance — Isolation Forest', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_FOLDER, 'shap_summary_plot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" SHAP Summary Plot saved: {path}")


def plot_shap_dot(shap_values, df_sample):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, df_sample, plot_type="dot", show=False)
    plt.title('SHAP Value Distribution per Feature', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_FOLDER, 'shap_dot_plot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" SHAP Dot Plot saved: {path}")


def plot_shap_waterfall(explainer, df_sample, res_sample):
    anomaly_indices = res_sample[res_sample['anomaly_if'] == 1].index.tolist()
    if len(anomaly_indices) == 0:
        print("  No anomalies in sample for waterfall plot.")
        return

    anomaly_idx = anomaly_indices[0]
    explanation = explainer(df_sample.iloc[[anomaly_idx]])

    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values        = explanation.values[0],
            base_values   = explanation.base_values[0],
            data          = df_sample.iloc[anomaly_idx].values,
            feature_names = df_sample.columns.tolist()
        ),
        show=False
    )
    plt.title(f'SHAP Waterfall Plot — Anomaly at Index {anomaly_idx}', fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    path = os.path.join(OUTPUT_FOLDER, 'shap_waterfall_plot.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" SHAP Waterfall Plot saved: {path}")


def plot_shap_force_matplotlib(explainer, df_sample, res_sample):
    """
    Generate SHAP force plot as a matplotlib figure.
    NOTE: shap.force_plot with matplotlib=True draws onto the current active
    figure — so we must call plt.gcf() AFTER shap draws, not create fig first.
    Returns a matplotlib figure, or None if no anomalies found.
    """
    import matplotlib.pyplot as plt

    anomaly_indices = res_sample[res_sample['anomaly_if'] == 1].index.tolist()
    if len(anomaly_indices) == 0:
        print("  No anomalies in sample for force plot.")
        return None

    anomaly_idx = anomaly_indices[0]
    shap_vals   = explainer.shap_values(df_sample.iloc[[anomaly_idx]])

    # Close any existing figures so SHAP draws on a clean canvas
    plt.close("all")

    shap.force_plot(
        explainer.expected_value,
        shap_vals[0],
        df_sample.iloc[anomaly_idx],
        matplotlib=True,
        show=False
    )

    # Capture whichever figure SHAP just drew on
    fig = plt.gcf()
    fig.set_size_inches(14, 3)
    plt.tight_layout()

    print(" SHAP Force Plot (matplotlib) generated.")
    return fig


def plot_shap_force(explainer, df_sample, res_sample):
    """Wrapper kept for backward compatibility when running standalone."""
    fig = plot_shap_force_matplotlib(explainer, df_sample, res_sample)
    if fig:
        path = os.path.join(OUTPUT_FOLDER, 'shap_force_plot.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f" SHAP Force Plot saved: {path}")


# ── kept for backward compatibility when running standalone ──────────────────

def run_shap_explainability(df_results=None, model_if=None, df_scaled=None, scaler=None):
    """
    Main function to run the SHAP explainability module.

    FIX: Now accepts df_results and model_if as parameters so the pipeline
    is NOT re-run when called from another module. Only runs the pipeline
    itself when called standalone (no arguments passed).
    """
    print(" Starting Module 6: SHAP Explainability")
    print("=" * 50)

    if df_results is None:
        df_results, model_if, _, _, df_scaled, scaler = run_noise_detection()

    df_sample, res_sample = get_shap_sample(df_scaled, df_results, SHAP_SAMPLE_SIZE)
    explainer, shap_values = compute_shap_values(model_if, df_sample)

    plot_shap_summary(shap_values, df_sample)
    plot_shap_dot(shap_values, df_sample)
    plot_shap_waterfall(explainer, df_sample, res_sample)
    plot_shap_force(explainer, df_sample, res_sample)

    print("\n Module 6 Complete: SHAP Explainability Successful!")
    return explainer, shap_values, df_sample, res_sample


if __name__ == "__main__":
    explainer, shap_values, df_sample, res_sample = run_shap_explainability()