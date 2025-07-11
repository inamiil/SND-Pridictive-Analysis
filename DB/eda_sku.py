import pandas as pd
from .db_connect import get_sales_data

def get_top_skus():
    df = get_sales_data()
    df = df[df['DIVISION'].isin(['CANDYLAND', 'BISCONNI'])]
    top = df.groupby(['SKU_CODE', 'SKU_LDESC'])['NET_SALE_AMOUNT'].sum().nlargest(10).reset_index()
    return top.to_dict(orient='records')

def get_pack_size_sales_by_division():
    df = get_sales_data()
    df = df[df['DIVISION'].isin(['CANDYLAND', 'BISCONNI'])]
    result = []

    for division in ['CANDYLAND', 'BISCONNI']:
        pack_sales = (
            df[df['DIVISION'] == division]
            .groupby('PACK_SIZE')['NET_SALE_AMOUNT']
            .sum().reset_index()
        )
        pack_sales['DIVISION'] = division
        result.extend(pack_sales.to_dict(orient='records'))

    return result

def get_top_brands_by_division():
    df = get_sales_data()
    df = df[df['DIVISION'].isin(['CANDYLAND', 'BISCONNI'])]
    result = []

    for division in ['CANDYLAND', 'BISCONNI']:
        top_brands = (
            df[df['DIVISION'] == division]
            .groupby('BRAND')['NET_SALE_AMOUNT']
            .sum().nlargest(10).reset_index()
        )
        top_brands['DIVISION'] = division
        result.extend(top_brands.to_dict(orient='records'))

    return result

def get_top_brand_packsize_combos():
    df = get_sales_data()
    df = df[df['DIVISION'].isin(['CANDYLAND', 'BISCONNI'])]
    result = []

    for division in ['CANDYLAND', 'BISCONNI']:
        combo = (
            df[df['DIVISION'] == division]
            .groupby(['BRAND', 'PACK_SIZE'])['NET_SALE_AMOUNT']
            .sum().nlargest(10).reset_index()
        )
        combo['DIVISION'] = division
        result.extend(combo.to_dict(orient='records'))

    return result

def get_top_flavours_by_division():
    df = get_sales_data()
    df = df[df['DIVISION'].isin(['CANDYLAND', 'BISCONNI'])]
    result = []

    for division in ['CANDYLAND', 'BISCONNI']:
        top_flavours = (
            df[df['DIVISION'] == division]
            .groupby('FLAVOUR')['NET_SALE_AMOUNT']
            .sum().nlargest(10).reset_index()
        )
        top_flavours['DIVISION'] = division
        result.extend(top_flavours.to_dict(orient='records'))

    return result
