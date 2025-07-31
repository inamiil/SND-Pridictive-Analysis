import pandas as pd
import numpy as np
from DB.db_connect import run_query
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def predict_sales(division: str = ""):
    # Base query
    query = """
    SELECT
        SUM(NET_SALE_AMOUNT) AS MONTHLY_SALES,
        REGION,
        YEAR,
        MONTH
    FROM FINAL_QUERY
    WHERE
        ((YEAR < 2025) OR (YEAR = 2025 AND MONTH < 5))
    """
   
    # Only add the division filter if it's not empty
    if division:
        query += f"AND DIVISION = '{division}'\n"
        query += """
        GROUP BY REGION, DIVISION, YEAR, MONTH
        ORDER BY REGION, DIVISION, YEAR, MONTH
        """
    else:
        query += """
        GROUP BY REGION, YEAR, MONTH
        ORDER BY REGION, YEAR, MONTH
        """
   
    df = pd.DataFrame(run_query(query))
   
    if df.empty:
        return {
            "summary_table": [],
            "forecast_table": [],
            "metrics": {},
            "regions": [],
        }
   
    # === Date Handling ===
    df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2) + "-01")
    df = df.sort_values(by=["REGION", "DATE"])
    df.drop(columns=["YEAR", "MONTH"], inplace=True)
   
    # === Feature Engineering ===
    df["LAG_1"] = df.groupby("REGION")["MONTHLY_SALES"].shift(1)
    df["LAG_2"] = df.groupby("REGION")["MONTHLY_SALES"].shift(2)
    df["LAG_3"] = df.groupby("REGION")["MONTHLY_SALES"].shift(3)
    df["ROLLING_MEAN_3"] = df.groupby("REGION")["MONTHLY_SALES"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["ROLLING_MEAN_6"] = df.groupby("REGION")["MONTHLY_SALES"].transform(lambda x: x.rolling(6, min_periods=1).mean())
    df["MONTH_NUM"] = df["DATE"].dt.month
   
    # === Add Seasonality Features ===
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH_NUM'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH_NUM'] / 12)
   
    # === Outlier Removal ===
    Q1 = df["MONTHLY_SALES"].quantile(0.1)
    Q3 = df["MONTHLY_SALES"].quantile(0.9)
    IQR = Q3 - Q1
    df = df[(df["MONTHLY_SALES"] >= Q1 - 1.5 * IQR) & (df["MONTHLY_SALES"] <= Q3 + 1.5 * IQR)]
   
    # === Drop NA & Log Transform ===
    df.dropna(subset=["LAG_1", "LAG_2", "LAG_3"], inplace=True)
    df["LOG_SALES"] = np.log1p(df["MONTHLY_SALES"])
   
    # === Ordinal Encoding REGION ===
    encoder = OrdinalEncoder()
    df["REGION_ENC"] = encoder.fit_transform(df[["REGION"]])
   
    # === Train/Test Split ===
    df['YEAR'] = df['DATE'].dt.year
    features = ["LAG_1", "LAG_2", "LAG_3", "ROLLING_MEAN_3", "ROLLING_MEAN_6", "MONTH_NUM", "MONTH_SIN", "MONTH_COS", "REGION_ENC"]
    train_df = df[df['YEAR'] < 2025]
    test_df = df[(df['YEAR'] == 2025) & (df['DATE'].dt.month <= 3)]
   
    X_train = train_df[features]
    y_train = train_df["LOG_SALES"]
    X_test = test_df[features]
    y_test = test_df["LOG_SALES"]
   
    # === Improved Grid Search ===
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.8, 1],
    }
    xgb = XGBRegressor(random_state=42, eval_metric='mae')
    grid_search = GridSearchCV(xgb, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
   
    # === Predict Jan-Apr 2025 ===
    log_preds = best_model.predict(X_test)
    preds = np.expm1(log_preds)
    actual = np.expm1(y_test)
   
    results_df = test_df.copy()
    results_df["PREDICTED_SALES"] = preds
    results_df["ERROR"] = results_df["MONTHLY_SALES"] - results_df["PREDICTED_SALES"]
   
    # === Forecast May-Dec 2025 (Fix Compounding Error) ===
    forecast_df = df.copy()
    forecast_months = pd.date_range("2025-05-01", "2025-12-01", freq="MS")
    all_forecasts = []
   
    for forecast_date in forecast_months:
        month_num = forecast_date.month
        sin_month = np.sin(2 * np.pi * month_num / 12)
        cos_month = np.cos(2 * np.pi * month_num / 12)
       
        next_inputs = []
        for region in forecast_df['REGION'].unique():
            region_data = forecast_df[forecast_df['REGION'] == region].sort_values("DATE")
            last_data = region_data.tail(6)
           
            if len(last_data) < 3:
                continue
           
            lag_1 = last_data.iloc[-1]['MONTHLY_SALES']
            lag_2 = last_data.iloc[-2]['MONTHLY_SALES']
            lag_3 = last_data.iloc[-3]['MONTHLY_SALES']
            rolling_3 = last_data['MONTHLY_SALES'].tail(3).mean()
            rolling_6 = last_data['MONTHLY_SALES'].mean()
            region_enc = encoder.transform([[region]])[0][0]
           
            row = {
                "LAG_1": lag_1,
                "LAG_2": lag_2,
                "LAG_3": lag_3,
                "ROLLING_MEAN_3": rolling_3,
                "ROLLING_MEAN_6": rolling_6,
                "MONTH_NUM": month_num,
                "MONTH_SIN": sin_month,
                "MONTH_COS": cos_month,
                "REGION_ENC": region_enc,
                "REGION": region,
                "DATE": forecast_date
            }
            next_inputs.append(row)
       
        input_df = pd.DataFrame(next_inputs)
        if input_df.empty:
            continue
       
        X_next = input_df[features]
        log_pred = best_model.predict(X_next)
        input_df["PREDICTED_SALES"] = np.expm1(log_pred)
        input_df["MONTHLY_SALES"] = input_df["PREDICTED_SALES"]  # Avoid error accumulation
       
        all_forecasts.append(input_df)
        forecast_df = pd.concat([forecast_df, input_df], ignore_index=True)
   
    # === Results Summary ===
    forecast_result = pd.concat(all_forecasts).sort_values(["REGION", "DATE"])
   
    # === Metrics ===
    train_preds = np.expm1(best_model.predict(X_train))
    train_actual = np.expm1(y_train)
    train_mae = mean_absolute_error(train_actual, train_preds)
    train_r2 = r2_score(train_actual, train_preds)
    test_mae = mean_absolute_error(actual, preds)
    test_r2 = r2_score(actual, preds)
 
    # Return the forecast results and metrics
    return {
        "summary_table": results_df[["REGION", "DATE", "MONTHLY_SALES", "PREDICTED_SALES", "ERROR"]].to_dict(orient="records"),
        "forecast_table": forecast_result[["REGION", "DATE", "PREDICTED_SALES"]].to_dict(orient="records"),
        "metrics": {
            "train_mae": round(train_mae, 2),
            "train_r2": round(train_r2, 2),
            "test_mae": round(test_mae, 2),
            "test_r2": round(test_r2, 2),
            "best_params": grid_search.best_params_
        },
        "regions": df["REGION"].unique().tolist()
    }