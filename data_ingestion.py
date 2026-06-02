import pandas as pd
import os
import sys

# Default dataset — used when no argument is passed
DEFAULT_DATASET_PATH = "iot_telemetry_data.csv"


def load_data(filepath):
    """Load any CSV dataset into a pandas DataFrame."""

    if not os.path.exists(filepath):
        print(f" Error: File '{filepath}' not found.")
        print(" Please make sure the CSV file is inside your project folder.")
        return None

    print(f" Loading dataset: {filepath}")
    try:
        df = pd.read_csv(filepath)
        print(f" Dataset loaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
        return df
    except Exception as e:
        print(f" Error reading file: {e}")
        return None


def inspect_data(df):
    """Perform basic inspection of the loaded dataset."""

    print("\n" + "=" * 50)
    print(" DATASET OVERVIEW")
    print("=" * 50)
    print(f"\n Total Rows    : {df.shape[0]}")
    print(f" Total Columns : {df.shape[1]}")

    print(f"\n Column Names:")
    for col in df.columns:
        print(f"   - {col}  ({df[col].dtype})")

    print(f"\n Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values found!")
    else:
        print(missing[missing > 0])

    print(f"\n First 5 Rows:")
    print(df.head())
    print("\n" + "=" * 50)


def run_ingestion(filepath=None):
    """
    Main function to run the data ingestion module.

    Usage from terminal:
        python data_ingestion.py                          # uses default IoT dataset
        python data_ingestion.py my_dataset.csv          # uses any CSV file

    Usage from another module:
        df, filepath = run_ingestion('my_dataset.csv')
    """

    print(" Starting Module 1: Data Ingestion")
    print("=" * 50)

    # Priority: argument passed > command line argument > default
    if filepath is None:
        if len(sys.argv) > 1:
            filepath = sys.argv[1]
            print(f" File path received from command line: {filepath}")
        else:
            filepath = DEFAULT_DATASET_PATH
            print(f" No file path given — using default: {filepath}")

    df = load_data(filepath)

    if df is not None:
        inspect_data(df)
        print("\n Module 1 Complete: Data Ingestion Successful!")
        return df, filepath
    else:
        print("\n Module 1 Failed: Could not load data.")
        return None, filepath


if __name__ == "__main__":
    df, filepath = run_ingestion()