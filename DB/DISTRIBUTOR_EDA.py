import pandas as pd
from .db_connect import get_sales_data

def get_top_distributors(n=10):
    df = get_sales_data()
    result = df.groupby('DISTRIBUTOR_NAME')['NET_SALE_AMOUNT'].sum().nlargest(n).reset_index()
    return result.to_dict(orient='records')

def get_yearly_sales_for_top_distributors(n=10):
    df = get_sales_data()
    top = df.groupby('DISTRIBUTOR_NAME')['NET_SALE_AMOUNT'].sum().nlargest(n).index.tolist()
    filtered = df[df['DISTRIBUTOR_NAME'].isin(top)]
    grouped = filtered.groupby(['YEAR', 'DISTRIBUTOR_NAME'])['NET_SALE_AMOUNT'].sum().reset_index()
    return grouped.to_dict(orient='records')

def get_division_vs_distributor_heatmap():
    df = get_sales_data()
    top10 = df.groupby('DISTRIBUTOR_NAME')['NET_SALE_AMOUNT'].sum().nlargest(10).index.tolist()
    filtered = df[df['DISTRIBUTOR_NAME'].isin(top10)]
    pivot = filtered.groupby(['DISTRIBUTOR_NAME', 'DIVISION'])['NET_SALE_AMOUNT'].sum().unstack().fillna(0).reset_index()
    return pivot.to_dict(orient='records')

def get_stacked_sales_by_distributor_division():
    df = get_sales_data()
    top5 = df.groupby('DISTRIBUTOR_NAME')['NET_SALE_AMOUNT'].sum().nlargest(5).index.tolist()
    pivot = df[df['DISTRIBUTOR_NAME'].isin(top5)]
    grouped = pivot.groupby(['DISTRIBUTOR_NAME', 'DIVISION'])['NET_SALE_AMOUNT'].sum().unstack().fillna(0).reset_index()
    return grouped.to_dict(orient='records')

def get_distributor_type_sales():
    df = get_sales_data()
    bisconni = set(df[df['DIVISION'] == 'BISCONNI']['DISTRIBUTOR_NAME'].unique())
    candyland = set(df[df['DIVISION'] == 'CANDYLAND']['DISTRIBUTOR_NAME'].unique())

    def classify(name):
        if name in bisconni and name in candyland:
            return 'Both'
        elif name in bisconni:
            return 'Only BISCONNI'
        else:
            return 'Only CANDYLAND'

    df['DISTRIBUTOR_TYPE'] = df['DISTRIBUTOR_NAME'].apply(classify)
    grouped = df.groupby('DISTRIBUTOR_TYPE')['NET_SALE_AMOUNT'].sum().reset_index()
    return grouped.to_dict(orient='records')
