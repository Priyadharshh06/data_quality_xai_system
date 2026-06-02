import sqlite3
import pandas as pd
import os
import sys
from data_ingestion import run_ingestion

DATABASE_NAME = "data_quality.db"


def get_table_name(filepath):
    """
    Derive a safe SQLite table name from the CSV filename.
    Example: 'iot_telemetry_data.csv' -> 'iot_telemetry_data'
             'sales_data_2024.csv'    -> 'sales_data_2024'
    """
    basename  = os.path.basename(filepath)           # e.g. iot_telemetry_data.csv
    tablename = os.path.splitext(basename)[0]        # remove .csv
    # Replace any characters that are not letters, digits, or underscore
    tablename = ''.join(c if c.isalnum() or c == '_' else '_' for c in tablename)
    return tablename.lower()


def create_connection():
    """Create a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_NAME)
    print(f" Connected to database: {DATABASE_NAME}")
    return conn


def create_table(conn, df, table_name):
    """
    Dynamically create a table based on the actual columns in the DataFrame.
    Table name is derived from the CSV filename — not hardcoded.
    """

    def get_sql_type(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return "INTEGER"
        elif pd.api.types.is_float_dtype(dtype):
            return "REAL"
        else:
            return "TEXT"

    col_definitions = ",\n        ".join(
        f"{col}  {get_sql_type(df[col].dtype)}" for col in df.columns
    )

    query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id  INTEGER PRIMARY KEY AUTOINCREMENT,
        {col_definitions}
    );
    """
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(query)
    conn.commit()
    print(f" Table '{table_name}' created dynamically based on dataset columns!")


def store_data(conn, df, table_name):
    """
    Store the DataFrame into the SQLite database.
    Converts boolean columns to integers for SQLite compatibility.
    """
    print(f"\n Storing data into table '{table_name}'...")

    df = df.copy()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df[col] = df[col].astype(int)
        print(f"   Converted boolean column '{col}' to integer.")

    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print(f" {len(df)} rows stored successfully!")


def verify_storage(conn, table_name):
    """Verify the data was stored correctly."""
    print(f"\n Verifying stored data in '{table_name}'...")
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
    count  = cursor.fetchone()[0]
    print(f" Total rows in database: {count}")

    print("\n Sample from database:")
    df_check = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5", conn)
    print(df_check)


def run_storage(filepath=None):
    """
    Main function to run the data storage module.

    Usage from terminal:
        python data_storage.py                      # uses default IoT dataset
        python data_storage.py my_dataset.csv       # uses any CSV file

    Usage from another module:
        conn, df, table_name = run_storage('my_dataset.csv')
    """

    print(" Starting Module 2: Data Storage")
    print("=" * 50)

    df, filepath = run_ingestion(filepath)

    if df is None:
        print(" Module 2 Failed: No data to store.")
        return None, None, None

    table_name = get_table_name(filepath)
    print(f" Table name derived from filename: '{table_name}'")

    conn = create_connection()
    create_table(conn, df, table_name)
    store_data(conn, df, table_name)
    verify_storage(conn, table_name)

    print("\n Module 2 Complete: Data Storage Successful!")
    return conn, df, table_name


if __name__ == "__main__":
    conn, df, table_name = run_storage()