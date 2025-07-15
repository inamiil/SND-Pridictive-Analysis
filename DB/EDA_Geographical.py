import pandas as pd
from .db_connect import get_geo_data

def get_daily_sales_by_month_for_territory(df, territory, division=None):
    df = get_geo_data()

    if df is None:
            raise ValueError("Dataframe is None — check your data loading function.")

    # ✅ Standardize case and types
    df['TERRITORY'] = df['TERRITORY'].astype(str).str.upper()
    if 'DIVISION' in df.columns:
        df['DIVISION'] = df['DIVISION'].astype(str).str.upper()

    df['NET_SALE_AMOUNT'] = pd.to_numeric(df['NET_SALE_AMOUNT'], errors='coerce')

    # ✅ Apply filtering
    df = df[df['TERRITORY'] == territory.upper()]

    # ✅ Filter by division if provided
    if division:
        df = df[df['DIVISION'] == division.upper()]  # <-- FILTER BY DIVISION HERE

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

            # Top/bottom 3 logic
            top_3 = grouped.nlargest(3, 'NET_SALE_AMOUNT')[['DAY', 'NET_SALE_AMOUNT', 'NET_SALE_AMOUNT_M']].to_dict(orient='records')
            bottom_3 = grouped.nsmallest(3, 'NET_SALE_AMOUNT')[['DAY', 'NET_SALE_AMOUNT', 'NET_SALE_AMOUNT_M']].to_dict(orient='records')

            key = f"{year}-{str(month).zfill(2)}"
            result[key] = {
                "data": grouped.to_dict(orient='records'),
                "top_3": top_3,
                "bottom_3": bottom_3
            }

    return result

