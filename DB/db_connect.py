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

 
def run_query(query):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
   
    # Get column names
    columns = [column[0] for column in cursor.description]
   
    # Fetch data
    print("Query executed, fetching data...")
    rows = cursor.fetchall()
   
    # Format as list of dictionaries
    result = [dict(zip(columns, row)) for row in rows]
   
    # Close connections
    cursor.close()
    conn.close()
    return result
 
