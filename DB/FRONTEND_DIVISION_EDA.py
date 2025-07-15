# === Imports ===
import pandas as pd
from .db_connect import get_sales_data
 
def get_region_sales():
    df = get_sales_data()
    return df.groupby('REGION')['NET_SALE_AMOUNT'].sum().reset_index().to_dict(orient='records')
 
def get_division_sales():
    df = get_sales_data()
    return df.groupby('DIVISION')['NET_SALE_AMOUNT'].sum().reset_index().to_dict(orient='records')
 
def get_yearly_sales_by_division():
    df = get_sales_data()
    filtered = df[df['DIVISION'].isin(['BISCONNI', 'CANDYLAND'])]
    grouped = filtered.groupby(['YEAR', 'DIVISION'])['NET_SALE_AMOUNT'].sum().reset_index()
    return grouped.to_dict(orient='records')
 
def get_monthly_sales_combined():
    df = get_sales_data()
    result = df.groupby(['YEAR', 'MONTH'])['NET_SALE_AMOUNT'].sum().reset_index()
    return result.to_dict(orient='records')
 
def get_monthly_sales_by_division():
    df = get_sales_data()
    filtered = df[df['DIVISION'].isin(['BISCONNI', 'CANDYLAND'])]
    grouped = filtered.groupby(['YEAR', 'MONTH', 'DIVISION'])['NET_SALE_AMOUNT'].sum().reset_index()
    return grouped.to_dict(orient='records')
 
def get_heatmap_data():
    df = get_sales_data()
    filtered = df[df['DIVISION'].isin(['BISCONNI', 'CANDYLAND'])]
    grouped = filtered.groupby(['DIVISION', 'YEAR', 'MONTH'])['NET_SALE_AMOUNT'].sum().reset_index()
    return grouped.to_dict(orient='records')