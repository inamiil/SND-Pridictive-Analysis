import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle

# Load the data
df = pd.read_csv("fake_sales_data.csv")

# Convert date column to datetime type
df['date'] = pd.to_datetime(df['date'])

# Extract new time-based features
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # Monday = 0, Sunday = 6

# Define features and target
X = df[['price', 'promotion', 'day', 'month', 'day_of_week']]
y = df['units_sold']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"\n📊 R² Score on test set: {r2:.4f}")

# Print sample predictions
print("\n🔍 Sample Predictions:")
for actual, predicted in zip(y_test[:5], y_pred[:5]):
    print(f"  Actual: {actual:.2f} | Predicted: {predicted:.2f}")

# Feature weights
print("\n📈 Feature weights:", model.coef_)
print("📉 Intercept:", model.intercept_)

# Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Units Sold")
plt.ylabel("Predicted Units Sold")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.tight_layout()
plt.show()
