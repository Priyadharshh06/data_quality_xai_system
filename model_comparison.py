import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def compare_anomaly_counts(df_results):
    """Compare how many anomalies each model detected."""

    print("\n Comparing anomaly counts across models...")

    total   = len(df_results)
    models  = ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM', 'Consensus (2/3)']
    columns = ['anomaly_if', 'anomaly_lof', 'anomaly_svm', 'anomaly_consensus']

    counts      = [df_results[col].sum() for col in columns]
    percentages = [round(c / total * 100, 2) for c in counts]

    comparison_df = pd.DataFrame({
        'Model': models,
        'Anomalies Detected': counts,
        'Percentage (%)': percentages
    })

    print("\n" + comparison_df.to_string(index=False))
    return comparison_df


def plot_anomaly_counts(comparison_df):
    """Plot a bar chart comparing anomaly counts."""

    print("\n Generating anomaly count comparison chart...")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    bars    = ax.bar(comparison_df['Model'],
                     comparison_df['Anomalies Detected'],
                     color=colors, edgecolor='black', width=0.5)

    for bar, pct in zip(bars, comparison_df['Percentage (%)']):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(comparison_df['Anomalies Detected']) * 0.02,
                f'{int(bar.get_height())}\n({pct}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Anomaly Detection Comparison Across Models',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Number of Anomalies Detected', fontsize=12)
    ax.set_ylim(0, max(comparison_df['Anomalies Detected']) * 1.2)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    path = os.path.join(OUTPUT_FOLDER, 'anomaly_count_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f" Chart saved: {path}")


def plot_agreement_matrix(df_results):
    """Plot a heatmap showing where models agree and disagree."""

    print("\n Generating model agreement heatmap...")

    agreement_df = pd.DataFrame({
        'Isolation Forest': df_results['anomaly_if'],
        'LOF':              df_results['anomaly_lof'],
        'One-Class SVM':    df_results['anomaly_svm']
    })

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(agreement_df.corr(), annot=True, fmt='.2f',
                cmap='coolwarm', linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Model Agreement Heatmap\n(Correlation of Anomaly Predictions)',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    path = os.path.join(OUTPUT_FOLDER, 'model_agreement_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f" Heatmap saved: {path}")


def plot_anomaly_distribution(df_results):
    """
    Plot distribution of anomalies per feature.
    FIX: Now uses actual columns from df_results instead of hardcoded IoT column names.
    """

    print("\n Generating anomaly distribution plots...")

    # Get only the data columns (exclude the anomaly label columns)
    feature_cols = [c for c in df_results.columns if not c.startswith('anomaly_')]

    n_cols = 3
    n_rows = -(-len(feature_cols) // n_cols)  # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, col in enumerate(feature_cols):
        normal  = df_results[df_results['anomaly_consensus'] == 0][col]
        anomaly = df_results[df_results['anomaly_consensus'] == 1][col]
        axes[i].hist(normal,  bins=50, alpha=0.6, color='#3498DB', label='Normal')
        axes[i].hist(anomaly, bins=50, alpha=0.6, color='#E74C3C', label='Anomaly')
        axes[i].set_title(f'{col.upper()} Distribution', fontweight='bold')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()

    # Hide any unused subplot slots
    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Normal vs Anomaly Distribution per Feature',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(OUTPUT_FOLDER, 'anomaly_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Distribution plot saved: {path}")


def run_model_comparison(df_results=None, model_if=None, model_lof=None,
                         model_svm=None, df_scaled=None, scaler=None):
    """
    Main function to run the model comparison module.

    FIX: Now accepts df_results as a parameter so the pipeline is NOT
    re-run when called from another module. Only runs the pipeline
    itself when called standalone (no arguments passed).
    """

    print(" Starting Module 5: Model Comparison")
    print("=" * 50)

    # Only run the full pipeline if no data was passed in
    if df_results is None:
        from noise_detection import run_noise_detection
        df_results, model_if, model_lof, model_svm, df_scaled, scaler = run_noise_detection()

    comparison_df = compare_anomaly_counts(df_results)
    plot_anomaly_counts(comparison_df)
    plot_agreement_matrix(df_results)
    plot_anomaly_distribution(df_results)

    print("\n Module 5 Complete: Model Comparison Successful!")
    print(f" All charts saved in the '{OUTPUT_FOLDER}' folder!")
    return df_results, model_if, model_lof, model_svm, df_scaled, scaler, comparison_df


if __name__ == "__main__":
    df_results, model_if, model_lof, model_svm, df_scaled, scaler, comparison_df = run_model_comparison()