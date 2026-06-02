import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def preprocess_dataframe(df, selected_cols):
    """
    Preprocess a DataFrame for noise detection.
    Accepts any DataFrame and a list of numerical column names.
    Returns: df_scaled, scaler, df_numerical
    """

    df_numerical = df[selected_cols].copy()

    if df_numerical.isnull().sum().sum() > 0:
        df_numerical = df_numerical.fillna(df_numerical.mean())

    df_numerical = df_numerical.drop_duplicates().reset_index(drop=True)

    scaler       = StandardScaler()
    scaled_array = scaler.fit_transform(df_numerical)
    df_scaled    = pd.DataFrame(scaled_array, columns=selected_cols)

    return df_scaled, scaler, df_numerical


# ── Standalone entry point ────────────────────────────────────────────────────

import sqlite3
import sys

DATABASE_NAME = "data_quality.db"


def get_numerical_columns(df):
    """Auto-detect numerical columns from a DataFrame."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def load_from_database(table_name):
    """Load data from SQLite database using the dynamic table name."""
    print(f" Loading data from table '{table_name}' in SQLite database...")
    conn = sqlite3.connect(DATABASE_NAME)
    df   = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    print(f" Loaded {len(df)} rows!")
    return df


def run_preprocessing(table_name=None):
    """
    Standalone entry point — loads from database and auto-detects numerical columns.

    Usage from terminal:
        python data_preprocessing.py                         # uses default IoT table
        python data_preprocessing.py sales_data_2024        # uses any table name
    """

    print(" Starting Module 3: Data Preprocessing")
    print("=" * 50)

    if table_name is None:
        if len(sys.argv) > 1:
            table_name = sys.argv[1]
        else:
            table_name = "iot_telemetry_data"   # default table name

    df               = load_from_database(table_name)
    numerical_cols   = get_numerical_columns(df)

    if len(numerical_cols) == 0:
        print(" No numerical columns found! Cannot preprocess.")
        return None, None, None

    print(f" Auto-detected numerical columns: {numerical_cols}")
    df_scaled, scaler, df_numerical = preprocess_dataframe(df, numerical_cols)

    print(f"\n PREPROCESSING SUMMARY")
    print(f" Rows processed : {len(df_numerical)}")
    print(f" Columns used   : {numerical_cols}")
    print(f" Scaling method : StandardScaler (mean=0, std=1)")
    print("\n Module 3 Complete: Data Preprocessing Successful!")

    return df_scaled, scaler, df_numerical


if __name__ == "__main__":
    df_scaled, scaler, df_numerical = run_preprocessing()