# === Imports ===
import pandas as pd
import numpy as np
from DB.db_connect import run_query
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# === Feature Preparation ===
def prepare_features(df):
    df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2) + "-01")
    df = df.sort_values(["REGION", "BRAND", "DATE"])
    df["NET_SALE_AMOUNT"] = pd.to_numeric(df["NET_SALE_AMOUNT"], errors="coerce").fillna(0)
    df.loc[df["NET_SALE_AMOUNT"] < 0, "NET_SALE_AMOUNT"] = 0

    all_dates = pd.date_range("2023-01-01", "2025-12-01", freq="MS")
    full_index = pd.MultiIndex.from_product([df["REGION"].unique(), df["BRAND"].unique(), all_dates],
                                            names=["REGION", "BRAND", "DATE"])
    df = df.set_index(["REGION", "BRAND", "DATE"]).reindex(full_index, fill_value=0).reset_index()

    df["MONTH_NUM"] = df["DATE"].dt.month
    df["YEAR_NUM"] = df["DATE"].dt.year
    df["LAG_1"] = df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(1)
    df["LAG_2"] = df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(2)
    df["LAG_3"] = df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(3)
    df["ROLLING_MEAN_3"] = df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["ROLLING_MEAN_6"] = df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH_NUM"] / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH_NUM"] / 12)

    Q1 = df["NET_SALE_AMOUNT"].quantile(0.01)
    Q3 = df["NET_SALE_AMOUNT"].quantile(0.99)
    IQR = Q3 - Q1
    df = df[(df["NET_SALE_AMOUNT"] >= Q1 - 1.5 * IQR) & (df["NET_SALE_AMOUNT"] <= Q3 + 1.5 * IQR)]
    df = df.dropna(subset=["LAG_1", "LAG_2", "LAG_3"])
    df["LOG_SALES"] = np.log1p(df["NET_SALE_AMOUNT"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    brand_encoder = LabelEncoder()
    df["BRAND_ENC"] = brand_encoder.fit_transform(df["BRAND"])
    return df, brand_encoder

# === Model Training ===
def train_xgboost_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1]
    }
    model = XGBRegressor(random_state=42, eval_metric='mae')
    grid = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_

# === Main Forecast Function ===
def generate_sales_forecast(var_region=None):
    # === Load Data ===
    if var_region:
        query = f"""
        SELECT REGION, BRAND, YEAR, MONTH, SUM(NET_SALE_AMOUNT) AS NET_SALE_AMOUNT
        FROM FINAL_QUERY
        WHERE REGION = '{var_region}'
        GROUP BY REGION, BRAND, YEAR, MONTH
        ORDER BY REGION, BRAND, YEAR, MONTH
        """
    else:
        query = """
        SELECT REGION, BRAND, YEAR, MONTH, SUM(NET_SALE_AMOUNT) AS NET_SALE_AMOUNT
        FROM FINAL_QUERY
        GROUP BY REGION, BRAND, YEAR, MONTH
        ORDER BY REGION, BRAND, YEAR, MONTH
        """

    raw_df = pd.DataFrame(run_query(query))

    # === Filter Top Brands ===
    if var_region:
        top_brands = raw_df.groupby("BRAND")['NET_SALE_AMOUNT'].sum().reset_index()
        print(f"Top brands sum: {top_brands}")
        top_3_brands = top_brands.sort_values("NET_SALE_AMOUNT", ascending=False).head(3)["BRAND"].tolist()
        print(f"Top 3 brands: {top_3_brands}")
        raw_df = raw_df[raw_df["BRAND"].isin(top_3_brands)]
    else:
        top_brands = raw_df.groupby(["REGION", "BRAND"])['NET_SALE_AMOUNT'].sum().reset_index()
        top_3_brands_per_region = top_brands.sort_values(["REGION", "NET_SALE_AMOUNT"], ascending=[True, False])
        top_3_brands_per_region = top_3_brands_per_region.groupby("REGION").head(3)
        raw_df = raw_df.merge(top_3_brands_per_region[["REGION", "BRAND"]], on=["REGION", "BRAND"])

    # === Feature Prep ===
    df, brand_encoder = prepare_features(raw_df)
    features = ["MONTH_NUM", "LAG_1", "LAG_2", "LAG_3", "ROLLING_MEAN_3", "ROLLING_MEAN_6", "BRAND_ENC", "MONTH_SIN", "MONTH_COS"]
    regions = [var_region] if var_region else df["REGION"].unique()
    combined_df_list = []
    actual_vs_predicted_list = []

    for region in regions:
        region_df = df[df["REGION"] == region]
        train = region_df[region_df["YEAR_NUM"] < 2025]
        test = region_df[(region_df["YEAR_NUM"] == 2025) & (region_df["DATE"].dt.month <= 4)]

        X_train = train[features]
        y_train = train["LOG_SALES"]
        best_model, _ = train_xgboost_model(X_train, y_train)

        actual_2024 = region_df[region_df["YEAR_NUM"] == 2024][["DATE", "REGION", "BRAND", "NET_SALE_AMOUNT"]].copy()
        actual_2024.rename(columns={"NET_SALE_AMOUNT": "SALES"}, inplace=True)
        actual_2024["TYPE"] = "Actual 2024"

        test = test.copy()
        test["SALES"] = np.expm1(best_model.predict(test[features])).round(0).astype(int)
        test["TYPE"] = "Predicted Jan-Apr 2025"

        region_forecast_df = region_df.copy()
        forecast_results = []
        for forecast_date in pd.date_range("2025-04-01", "2025-12-01", freq="MS"):
            temp_df = region_forecast_df[region_forecast_df["DATE"] == forecast_date].copy()
            if temp_df.empty:
                continue
            predicted_log_sales = best_model.predict(temp_df[features])
            predicted_sales = np.expm1(predicted_log_sales).round(0).astype(int)
            temp_df["SALES"] = predicted_sales
            temp_df["TYPE"] = "Forecast Apr-Dec 2025"
            forecast_results.append(temp_df[["DATE", "REGION", "BRAND", "SALES", "TYPE"]])

            region_forecast_df.loc[region_forecast_df["DATE"] == forecast_date, "NET_SALE_AMOUNT"] = predicted_sales
            region_forecast_df["LAG_1"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(1)
            region_forecast_df["LAG_2"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(2)
            region_forecast_df["LAG_3"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].shift(3)
            region_forecast_df["ROLLING_MEAN_3"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(3, min_periods=1).mean())
            region_forecast_df["ROLLING_MEAN_6"] = region_forecast_df.groupby(["REGION", "BRAND"])["NET_SALE_AMOUNT"].transform(lambda x: x.rolling(6, min_periods=1).mean())

        forecast_df = pd.concat(forecast_results)
        combined_df = pd.concat([actual_2024, test[["DATE", "REGION", "BRAND", "SALES", "TYPE"]], forecast_df])
        combined_df_list.append(combined_df)

        # Actual vs Predicted Jan–Apr 2025
        for brand in df["BRAND"].unique():
            brand_test = test[test["BRAND"] == brand]
            actual_2025 = region_df[
                (region_df["BRAND"] == brand) &
                (region_df["YEAR_NUM"] == 2025) &
                (region_df["DATE"].dt.month <= 4)
            ][["DATE", "NET_SALE_AMOUNT"]].rename(columns={"NET_SALE_AMOUNT": "ACTUAL"})
            merged = brand_test.merge(actual_2025, on="DATE", how="left")
            merged["BRAND"] = brand
            merged["REGION"] = region
            actual_vs_predicted_list.append(merged[["DATE", "REGION", "BRAND", "ACTUAL", "SALES"]])

    combined_all = pd.concat(combined_df_list)
    actual_vs_predicted_all = pd.concat(actual_vs_predicted_list)

    combined_all = pd.concat(combined_df_list)
 
    # === National-Level Combined Sales ===
    combined_national = combined_all.groupby(["BRAND", "DATE", "TYPE"])["SALES"].sum().reset_index()
    national_sales_records = combined_national.to_dict(orient="records")  # Ready for frontend
        # === Return as API Response ===
        
    return {
        "summary": combined_all.to_dict(orient="records"),
        "actual_vs_predicted": actual_vs_predicted_all.to_dict(orient="records"),
        "brands": sorted(df["BRAND"].unique().tolist()),
        "regions": sorted(regions)
    }
