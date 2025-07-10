# db_connect.py
import pyodbc
import pandas as pd

def get_connection():
    return pyodbc.connect(
        'DRIVER={ODBC Driver 18 for SQL Server};'
        'SERVER=172.19.0.75\\sndpro;'
        'DATABASE=Project;'
        'UID=project;'
        'PWD=P@kistan!@#$;'
        'Encrypt=no;'
    )

def run_query(query):
    try:
        conn = get_connection()
        df = pd.read_sql(query, conn)  # Much easier & safer than manual cursor
        conn.close()
        return df
    except Exception as e:
        print("❌ Error running query:", e)
        return pd.DataFrame()  # Return empty DataFrame on failure
