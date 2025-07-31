# Updated Forecasting Script with Improved Forecast Loop, Grid Search, and Seasonality
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from db_connect import run_query
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
import matplotlib.ticker as ticker

# === Load Data ===
query = """
SELECT 
    SUM(NET_SALES_UNITS) AS MONTHLY_SALES,
    REGION,
    YEAR,
    MONTH
FROM FINAL_QUERY
WHERE 
    (YEAR < 2025) OR (YEAR = 2025 AND MONTH < 4)
GROUP BY REGION, YEAR, MONTH
ORDER BY REGION, YEAR, MONTH
"""

df = pd.DataFrame(run_query(query))

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
grid_search = GridSearchCV(xgb, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)
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
forecast_months = pd.date_range("2025-04-01", "2025-12-01", freq="MS")
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
    X_next = input_df[features]
    log_pred = best_model.predict(X_next)
    input_df["PREDICTED_SALES"] = np.expm1(log_pred)
    input_df["MONTHLY_SALES"] = input_df["PREDICTED_SALES"]  # Avoid error accumulation

    all_forecasts.append(input_df)
    forecast_df = pd.concat([forecast_df, input_df], ignore_index=True)

# === Results Summary ===
forecast_result = pd.concat(all_forecasts).sort_values(["REGION", "DATE"])
print("\n===  Actual vs Predicted Sales (Jan–Apr 2025) ===")
print(results_df[["REGION", "DATE", "MONTHLY_SALES", "PREDICTED_SALES", "ERROR"]].to_string(index=False))
print("\n===  Forecasted Sales (May–Dec 2025) ===")
print(forecast_result[["REGION", "DATE", "PREDICTED_SALES"]].to_string(index=False))

# === Metrics ===
print(f"\nTraining MAE: {mean_absolute_error(np.expm1(y_train), np.expm1(best_model.predict(X_train))):.2f}")
print(f"Training R² Score: {r2_score(np.expm1(y_train), np.expm1(best_model.predict(X_train))):.2f}")
print(f"Best Params: {grid_search.best_params_}")
print(f"Test MAE: {mean_absolute_error(actual, preds):.2f}")
print(f"Test R² Score: {r2_score(actual, preds):.2f}")

# === Visualizations ===
import seaborn as sns
sns.set_style("whitegrid")

# === 1. Actual vs Predicted Plot (Jan–Apr 2025) ===
# === Define Consistent Colors for Regions ===
region_colors = {
    "SOUTH": "blue",
    "NORTH": "orange"
}

plt.figure(figsize=(14, 6))
for region, group in results_df.groupby("REGION"):
    color = region_colors.get(region, "black")  # Fallback to black if region not in dict
    plt.plot(group["DATE"], group["MONTHLY_SALES"], label=f"{region} (Actual)", marker='o', color=color)
    plt.plot(group["DATE"], group["PREDICTED_SALES"], label=f"{region} (Predicted)", linestyle='--', marker='x', color=color)



plt.title(" Actual vs Predicted Monthly Sales (Jan–Apr 2025)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Sales Units")
plt.legend()
plt.grid(True)
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1_000_000:.1f}M' if x >= 1_000_000 else f'{x/1_000:.0f}K'))
plt.tight_layout()
plt.show()

# === 2. Forecasted Sales Plot (May–Dec 2025) ===
for region, group in forecast_result.groupby("REGION"):
    color = region_colors.get(region, "black")
    plt.figure(figsize=(12, 5))
    plt.plot(group["DATE"], group["PREDICTED_SALES"], marker='o', linestyle='-', color=color)

    plt.title(f"📊= Forecasted Monthly Sales for {region} (May–Dec 2025)", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Sales Units")
    plt.grid(True)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1_000_000:.1f}M' if x >= 1_000_000 else f'{x/1_000:.0f}K'))
    plt.tight_layout()
    plt.show()

# === 3. Combined Full Timeline Plot (2023–2025 Actual + Forecasted) ===
combined_df = pd.concat([df, forecast_result], ignore_index=True)
for region, group in combined_df.groupby("REGION"):
    plt.figure(figsize=(14, 6))
    group_sorted = group.sort_values("DATE")
    color = region_colors.get(region, "black")
    plt.plot(group_sorted["DATE"], group_sorted["MONTHLY_SALES"], marker='o', linestyle='-', label=f"{region}", color=color)
    plt.axvline(pd.to_datetime("2025-04-30"), color='red', linestyle='--', label="Forecast Start")
    plt.title(f"📈 Full Sales Timeline for {region} (2023–2025)", fontsize=14)
    plt.xlabel("Date")
    plt.ylabel("Monthly Sales Units")
    plt.grid(True)
    plt.legend()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1_000_000:.1f}M' if x >= 1_000_000 else f'{x/1_000:.0f}K'))
    plt.tight_layout()
    plt.show()
