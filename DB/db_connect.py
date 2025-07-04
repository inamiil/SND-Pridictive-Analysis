import pyodbc

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
    
    columns = [column[0] for column in cursor.description]
    rows = cursor.fetchall()
    
    result = [dict(zip(columns, row)) for row in rows]
    
    cursor.close()
    conn.close()
    return result
