# edas.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from db_connect import get_sales_data

# --- LOAD DATA ---
df = get_sales_data()

# ---- EDA ---
print("📌 Sample Data:")
print(df.head())

print("\n📦 Shape of data:", df.shape)

print("\n🧱 Null values:")
print(df.isnull().sum())

print("\n🔠 Data types:")
print(df.dtypes)

print("\n📊 Summary statistics:")
print(df.describe())

print("\n🏷️ DIVISIONS by Total Sale Amount:")
print(df.groupby('DIVISION')["NET_SALE_AMOUNT"].sum().sort_values(ascending=False).head(10))

print("\n🌍 Sales by Region:")
print(df.groupby('REGION')["NET_SALE_AMOUNT"].sum())

print("\n🗓️ Sales by Month:")
print(df.groupby('MONTH')["NET_SALE_AMOUNT"].sum().sort_index())

# --- BISCONNI Yearly Sales ---
bisconni_df = df[df['DIVISION'] == 'BISCONNI']
yearly_sales = bisconni_df.groupby('YEAR')['NET_SALE_AMOUNT'].sum().sort_index()
print("📈 Yearly Sales for BISCONNI:")
print(yearly_sales)

# --- BISCONNI Monthly Sales ---
monthly_sales = bisconni_df.groupby('MONTH')['NET_SALE_AMOUNT'].sum().sort_index()
print("\n📊 Monthly Sales for BISCONNI:")
print(monthly_sales)

# ---- Visuals ----

# 📊 Monthly Sales (All Brands)
df.groupby('MONTH')["NET_SALE_AMOUNT"].sum().sort_index().plot(
    kind='bar',
    title='Monthly Sales',
    xlabel='Month',
    ylabel='Net Sale Amount',
    figsize=(10, 5),
    color='skyblue'
)
plt.tight_layout()
plt.show()

# 📊 Region Sales
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='REGION', y='NET_SALE_AMOUNT', estimator=sum, ci=None)
plt.title('Sales by Region')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📊 Division Sales
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='DIVISION', y='NET_SALE_AMOUNT', estimator=sum, ci=None)
plt.title('Sales by Division')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 📈 Yearly Sales Comparison: BISCONNI vs CANDYLAND
filtered_df = df[df['DIVISION'].str.upper().isin(['BISCONNI', 'CANDYLAND'])]
yearly_sales = (
    filtered_df.groupby(['YEAR', 'DIVISION'])['NET_SALE_AMOUNT']
    .sum()
    .reset_index()
)

plt.figure(figsize=(10, 5))
sns.barplot(data=yearly_sales, x='YEAR', y='NET_SALE_AMOUNT', hue='DIVISION')
plt.title("📈 Yearly Sales: BISCONNI vs CANDYLAND")
plt.ylabel("Net Sales")
plt.xlabel("Year")
plt.legend(title='Division')
plt.tight_layout()
plt.show()

# 📊 Monthly Sales Comparison
monthly_sales = (
    filtered_df.groupby(['MONTH', 'DIVISION'])['NET_SALE_AMOUNT']
    .sum()
    .reset_index()
)

plt.figure(figsize=(10, 5))
sns.barplot(data=monthly_sales, x='MONTH', y='NET_SALE_AMOUNT', hue='DIVISION')
plt.title("📊 Monthly Sales (All Years): BISCONNI vs CANDYLAND")
plt.ylabel("Net Sales")
plt.xlabel("Month")
plt.legend(title='Division')
plt.tight_layout()
plt.show()

# 🌡️ Heatmaps for Monthly Sales by Year
heatmap_data = (
    filtered_df
    .groupby(['DIVISION', 'YEAR', 'MONTH'])['NET_SALE_AMOUNT']
    .sum()
    .reset_index()
)

for division in heatmap_data['DIVISION'].unique():
    pivot = heatmap_data[heatmap_data['DIVISION'] == division].pivot_table(
        index='YEAR',
        columns='MONTH',
        values='NET_SALE_AMOUNT',
        aggfunc='sum'
    ).fillna(0)

    pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]

    cmap = 'YlGnBu' if division == 'BISCONNI' else 'OrRd'

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap=cmap)
    plt.title(f"🌡️ Monthly Sales Heatmap - {division}")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.show()
