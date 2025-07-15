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

def get_sales_data():
    query = """
        SELECT TOP 10000 *
        FROM [Project].[dbo].[FINAL_QUERY]
    """
    return run_query(query, as_dataframe=True)

def get_national_brands_data():
    query = """
    WITH BrandSales AS (
        SELECT 
            BRAND,
            YEAR,
            MONTH,
            SUM(NET_SALE_AMOUNT) AS MONTHLY_SALES
        FROM FINAL_QUERY
        GROUP BY BRAND, YEAR, MONTH
    ),
    TopBrands AS (
        SELECT TOP 3 BRAND
        FROM FINAL_QUERY
        GROUP BY BRAND
        ORDER BY SUM(NET_SALE_AMOUNT) DESC
    )
    SELECT 
        BS.BRAND,
        BS.YEAR,
        BS.MONTH,
        BS.MONTHLY_SALES
    FROM BrandSales BS
    JOIN TopBrands TB ON BS.BRAND = TB.BRAND
    ORDER BY BS.BRAND, BS.YEAR, BS.MONTH;
    """
    return run_query(query, as_dataframe=False)


def get_regional_brands_data():
    query = """
    WITH RankedBrands AS (
        SELECT 
            REGION,
            BRAND,
            YEAR,
            MONTH,
            SUM(NET_SALE_AMOUNT) AS MONTHLY_SALES
        FROM FINAL_QUERY
        WHERE BRAND IS NOT NULL
        GROUP BY REGION, BRAND, YEAR, MONTH
    ),
    TopBrands AS (
        SELECT REGION, BRAND
        FROM (
            SELECT 
                REGION, 
                BRAND,
                SUM(MONTHLY_SALES) AS TOTAL_SALES,
                ROW_NUMBER() OVER (PARTITION BY REGION ORDER BY SUM(MONTHLY_SALES) DESC) AS RN
            FROM RankedBrands
            GROUP BY REGION, BRAND
        ) AS Sub
        WHERE RN <= 3
    ),
    MonthlyTrend AS (
        SELECT 
            F.REGION,
            F.BRAND,
            F.YEAR,
            F.MONTH,
            SUM(F.NET_SALE_AMOUNT) AS MONTHLY_SALES
        FROM FINAL_QUERY F
        INNER JOIN TopBrands T
            ON F.REGION = T.REGION AND F.BRAND = T.BRAND
        GROUP BY F.REGION, F.BRAND, F.YEAR, F.MONTH
    )
    SELECT * 
    FROM MonthlyTrend
    ORDER BY REGION, BRAND, YEAR, MONTH;
    """
    return run_query(query, as_dataframe=False)

def get_geo_data():
    query = """
    SELECT 10
    DATE_ID,
    DAY,
    MONTH,
    YEAR,
    DIVISION,
    TERRITORY,
    NET_SALE_AMOUNT
    FROM FINAL_QUERY
    """
    return run_query(query, as_dataframe=False)

def run_query(query, as_dataframe=False):
    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    return df if as_dataframe else df.to_dict(orient='records')
