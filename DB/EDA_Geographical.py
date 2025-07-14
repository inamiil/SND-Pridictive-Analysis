import pandas as pd
from .db_connect import get_sales_data, run_query

# EDA_GEOGRAPHICAL.py

def get_daily_sales_by_month_for_territory(df, territory):
    """
    Prepares daily sales by month for a territory.
    Returns a dictionary like:
    {
        "2024-01": [{"DAY": 1, "NET_SALE_AMOUNT": 100000, "NET_SALE_AMOUNT_M": 0.1}, ...],
        ...
    }
    """
    df['TERRITORY'] = df['TERRITORY'].astype(str).str.upper()
    df['NET_SALE_AMOUNT'] = pd.to_numeric(df['NET_SALE_AMOUNT'], errors='coerce')
    df = df[df['TERRITORY'] == territory.upper()]

    if df.empty:
        return {}

    result = {}

    for year in sorted(df['YEAR'].unique()):
        year_df = df[df['YEAR'] == year]
        for month in sorted(year_df['MONTH'].unique()):
            month_df = year_df[year_df['MONTH'] == month].copy()
            grouped = (
                month_df.groupby('DAY')['NET_SALE_AMOUNT']
                .sum()
                .sort_index()
                .reset_index()
            )
            grouped['NET_SALE_AMOUNT_M'] = grouped['NET_SALE_AMOUNT'] / 1e6

            key = f"{year}-{str(month).zfill(2)}"
            result[key] = grouped.to_dict(orient='records')

    return result
