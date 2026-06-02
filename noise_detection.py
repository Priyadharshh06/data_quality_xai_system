import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
import time
from data_preprocessing import run_preprocessing


def run_isolation_forest(df_scaled, contamination=0.05):
    """Run Isolation Forest anomaly detection."""
    print("\n Running Isolation Forest...")
    start = time.time()

    model_if = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    predictions = model_if.fit_predict(df_scaled)
    labels = [1 if p == -1 else 0 for p in predictions]

    elapsed = round(time.time() - start, 2)
    print(f" Isolation Forest complete in {elapsed}s — {sum(labels)} anomalies detected")
    return labels, model_if


def run_lof(df_scaled, contamination=0.05):
    """Run Local Outlier Factor anomaly detection."""
    print("\n Running Local Outlier Factor (LOF)...")
    start = time.time()

    model_lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    predictions = model_lof.fit_predict(df_scaled)
    labels = [1 if p == -1 else 0 for p in predictions]

    elapsed = round(time.time() - start, 2)
    print(f" LOF complete in {elapsed}s — {sum(labels)} anomalies detected")
    return labels, model_lof


def run_one_class_svm(df_scaled, contamination=0.05):
    """Run One-Class SVM anomaly detection. Trains on a sample for performance."""
    print("\n Running One-Class SVM...")
    start = time.time()

    # FIX: safely cap sample size to actual dataset size
    sample_size = min(10000, len(df_scaled))
    df_sample = df_scaled.sample(n=sample_size, random_state=42)

    model_svm = OneClassSVM(kernel='rbf', nu=contamination)
    model_svm.fit(df_sample)

    full_predictions = model_svm.predict(df_scaled)
    labels = [1 if p == -1 else 0 for p in full_predictions]

    elapsed = round(time.time() - start, 2)
    print(f" One-Class SVM complete in {elapsed}s — {sum(labels)} anomalies detected")
    return labels, model_svm


def combine_results(df_numerical, if_labels, lof_labels, svm_labels):
    """Combine all three model results into a single DataFrame."""
    df_results = df_numerical.copy()
    df_results['anomaly_if']  = if_labels
    df_results['anomaly_lof'] = lof_labels
    df_results['anomaly_svm'] = svm_labels
    df_results['anomaly_consensus'] = (
        (df_results['anomaly_if'] +
         df_results['anomaly_lof'] +
         df_results['anomaly_svm']) >= 2
    ).astype(int)
    return df_results


def detect_anomalies(df_scaled, df_numerical, contamination=0.05):
    """
    Main reusable function for noise detection.
    Accepts preprocessed df_scaled and original df_numerical.
    Returns: df_results, model_if, model_lof, model_svm
    """
    if_labels,  model_if  = run_isolation_forest(df_scaled, contamination)
    lof_labels, model_lof = run_lof(df_scaled, contamination)
    svm_labels, model_svm = run_one_class_svm(df_scaled, contamination)

    df_results = combine_results(df_numerical, if_labels, lof_labels, svm_labels)

    print("\n NOISE DETECTION SUMMARY")
    print(f"Total points       : {len(df_results)}")
    print(f"Isolation Forest   : {df_results['anomaly_if'].sum()} anomalies")
    print(f"LOF                : {df_results['anomaly_lof'].sum()} anomalies")
    print(f"One-Class SVM      : {df_results['anomaly_svm'].sum()} anomalies")
    print(f"Consensus (2/3)    : {df_results['anomaly_consensus'].sum()} anomalies")

    return df_results, model_if, model_lof, model_svm


# ── kept for backward compatibility when running standalone ──────────────────

def run_noise_detection():
    """Standalone entry point — pulls data from database pipeline."""
    print(" Starting Module 4: Noise Detection")
    print("=" * 50)

    df_scaled, scaler, df_numerical = run_preprocessing()
    df_results, model_if, model_lof, model_svm = detect_anomalies(df_scaled, df_numerical)

    print("\n Module 4 Complete: Noise Detection Successful!")
    return df_results, model_if, model_lof, model_svm, df_scaled, scaler


if __name__ == "__main__":
    df_results, model_if, model_lof, model_svm, df_scaled, scaler = run_noise_detection()