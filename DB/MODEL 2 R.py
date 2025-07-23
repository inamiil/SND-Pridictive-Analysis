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
GROUP BY REGION, YEAR, MONTH
ORDER BY REGION, YEAR, MONTH
"""
df = pd.DataFrame(run_query(query))

# === Date Handling ===
df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str).str.zfill(2) + "-01")
df.drop(columns=["MONTH", "YEAR"], inplace=True)
df = df.sort_values(by=["REGION", "DATE"])

# === Lag & Rolling Features ===
df["LAG_1"] = df.groupby("REGION")["MONTHLY_SALES"].shift(1)
df["LAG_2"] = df.groupby("REGION")["MONTHLY_SALES"].shift(2)
df["LAG_3"] = df.groupby("REGION")["MONTHLY_SALES"].shift(3)
df["ROLLING_MEAN_3"] = df.groupby("REGION")["MONTHLY_SALES"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["ROLLING_MEAN_6"] = df.groupby("REGION")["MONTHLY_SALES"].transform(lambda x: x.rolling(6, min_periods=1).mean())
df["MONTH_NUM"] = df["DATE"].dt.month
df["YEAR_NUM"] = df["DATE"].dt.year

# === Outlier Removal ===
Q1 = df["MONTHLY_SALES"].quantile(0.1)
Q3 = df["MONTHLY_SALES"].quantile(0.9)
IQR = Q3 - Q1
df = df[(df["MONTHLY_SALES"] >= Q1 - 1.5 * IQR) & (df["MONTHLY_SALES"] <= Q3 + 1.5 * IQR)]

# === Drop NA & Log Transform ===
df = df.dropna(subset=["LAG_1", "LAG_2", "LAG_3"])
df["LOG_SALES"] = np.log1p(df["MONTHLY_SALES"])

# === Ordinal Encoding REGION ===
encoder = OrdinalEncoder()
df["REGION_ENC"] = encoder.fit_transform(df[["REGION"]])

# === Train/Test Split ===
features = ["LAG_1", "LAG_2", "LAG_3", "ROLLING_MEAN_3", "ROLLING_MEAN_6", "MONTH_NUM", "REGION_ENC"]
train_df = df[(df["YEAR_NUM"] == 2023) | (df["YEAR_NUM"] == 2024)]
test_df = df[(df["YEAR_NUM"] == 2025) & (df["DATE"].dt.month <= 4)]

X_train = train_df[features]
y_train = train_df["LOG_SALES"]
X_test = test_df[features]
y_test = test_df["LOG_SALES"]

# === TimeSeriesSplit + Grid Search ===
param_grid = {
    'n_estimators': [100],
    'max_depth': [4],
    'learning_rate': [0.05],
    'subsample': [1,0],
    'colsample_bytree': [0.8],
    'gamma': [0],
    'reg_alpha': [0],
    'reg_lambda': [0]
}

xgb = XGBRegressor(random_state=42, eval_metric='mae')
grid_search = GridSearchCV(xgb, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# === Predict Jan–Apr 2025 ===
log_preds = best_model.predict(X_test)
preds = np.expm1(log_preds)
actual = np.expm1(y_test)

mae = mean_absolute_error(actual, preds)
r2 = r2_score(actual, preds)

print(f"Best Params: {grid_search.best_params_}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

results_df = test_df.copy()
results_df["PREDICTED_SALES"] = preds

# === Plot Actual vs Predicted ===
plt.figure(figsize=(14, 6))
for key, grp in results_df.groupby("REGION"):
    plt.plot(grp["DATE"], grp["MONTHLY_SALES"], label=f"{key} (Actual)", marker='o')
    plt.plot(grp["DATE"], grp["PREDICTED_SALES"], label=f"{key} (Predicted)", linestyle='--', marker='x')
plt.title("📈 Actual vs Predicted Sales (Jan–Apr 2025) - XGBoost")
plt.xlabel("Date")
plt.ylabel("Monthly Sales Units")
plt.legend()
plt.grid(True)
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1_000_000:.1f}M' if x >= 1_000_000 else f'{x/1_000:.0f}K'))
plt.tight_layout()
plt.show()

# === Forecast May–Dec 2025 ===
forecast_df = df.copy()
future_months = pd.date_range(start="2025-05-01", end="2025-12-01", freq="MS")
all_forecasts = []

for forecast_date in future_months:
    forecast_month = forecast_date.month
    forecast_year = forecast_date.year
    next_input = []

    for region in forecast_df["REGION"].unique():
        region_df = forecast_df[forecast_df["REGION"] == region].sort_values("DATE")
        last_3 = region_df.tail(3)
        if len(last_3) < 3:
            continue

        lag_1 = last_3.iloc[-1]["MONTHLY_SALES"]
        lag_2 = last_3.iloc[-2]["MONTHLY_SALES"]
        lag_3 = last_3.iloc[-3]["MONTHLY_SALES"]
        rolling_mean_3 = region_df["MONTHLY_SALES"].tail(3).mean()
        rolling_mean_6 = region_df["MONTHLY_SALES"].tail(6).mean()
        region_enc = encoder.transform([[region]])[0][0]

        next_input.append({
            "REGION": region,
            "LAG_1": lag_1,
            "LAG_2": lag_2,
            "LAG_3": lag_3,
            "ROLLING_MEAN_3": rolling_mean_3,
            "ROLLING_MEAN_6": rolling_mean_6,
            "MONTH_NUM": forecast_month,
            "REGION_ENC": region_enc,
            "DATE": forecast_date
        })

    input_df = pd.DataFrame(next_input)
    if input_df.empty:
        continue

    X_next = input_df[features]
    log_pred_next = best_model.predict(X_next)
    pred_sales = np.expm1(log_pred_next)

    input_df["PREDICTED_SALES"] = pred_sales
    input_df["MONTHLY_SALES"] = pred_sales
    all_forecasts.append(input_df)
    forecast_df = pd.concat([forecast_df, input_df], ignore_index=True)

# === Forecast Plot ===
forecast_result = pd.concat(all_forecasts).sort_values(["REGION", "DATE"])
for region, group in forecast_result.groupby("REGION"):
    plt.figure(figsize=(12, 5))
    plt.plot(group["DATE"], group["PREDICTED_SALES"], marker='o', linestyle='-', color='tab:blue')
    plt.title(f"📊 Forecasted Monthly Sales for {region} (May–Dec 2025)", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Sales Units")
    plt.grid(True)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1_000_000:.1f}M' if x >= 1_000_000 else f'{x/1_000:.0f}K'))
    plt.tight_layout()
    plt.show()

# === Print Results ===
print("\n=== 📌 Actual vs Predicted Sales (Jan–Apr 2025) ===")
summary = results_df[["REGION", "DATE", "MONTHLY_SALES", "PREDICTED_SALES"]].copy()
summary["ERROR"] = summary["MONTHLY_SALES"] - summary["PREDICTED_SALES"]
print(summary.sort_values(["REGION", "DATE"]).to_string(index=False))

print("\n=== 📌 Forecasted Sales (May–Dec 2025) ===")
forecast_summary = forecast_result[["REGION", "DATE", "PREDICTED_SALES"]].copy()
forecast_summary = forecast_summary.sort_values(["REGION", "DATE"])
print(forecast_summary.to_string(index=False))

# === Training Metrics ===
train_preds_log = best_model.predict(X_train)
train_preds = np.expm1(train_preds_log)
train_actual = np.expm1(y_train)

print(f"\nTraining MAE: {mean_absolute_error(train_actual, train_preds):.2f}")
print(f"Training R² Score: {r2_score(train_actual, train_preds):.2f}")

mae = mean_absolute_error(actual, preds)
r2 = r2_score(actual, preds)

print(f"Best Params: {grid_search.best_params_}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")