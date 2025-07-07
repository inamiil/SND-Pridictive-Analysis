# db_connect.py
import pyodbc
import pandas as pd

def get_connection():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 18 for SQL Server};'
        'SERVER=172.19.0.75\\sndpro;'
        'DATABASE=Project;'
        'UID=project;'
        'PWD=P@kistan!@#$;'
        'Encrypt=no;'
    )
    return conn

def get_sales_data():
    conn = get_connection()
    query = """
        SELECT TOP 10000 *
        FROM [Project].[dbo].[FINAL_QUERY]
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df
