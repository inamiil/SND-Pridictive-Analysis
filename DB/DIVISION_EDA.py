import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import matplotlib.ticker as ticker
from db_connect import get_sales_data

# === Load Data ===
df = get_sales_data()

# === Set Colors ===
BISCONNI_COLOR = '#1f77b4'   # Blue
CANDYLAND_COLOR = '#ff7f0e' # Orange
COMBINED_COLOR = '#7851A9'  # Brownish blend

# === Quick EDA Summary ===
print("\U0001F4CC Sample Data:\n", df.head())
print("\n\U0001F4E6 Shape:", df.shape)
print("\n\U0001F9F1 Nulls:\n", df.isnull().sum())
print("\n\U0001F520 Dtypes:\n", df.dtypes)
print("\n\U0001F4CA Describe:\n", df.describe())

# === Grouped Stats ===
print("\n\U0001F3F7️ Top Divisions by Sales:\n", df.groupby('DIVISION')["NET_SALE_AMOUNT"].sum().sort_values(ascending=False))
print("\n\U0001F30D Region-wise Sales:\n", df.groupby('REGION')["NET_SALE_AMOUNT"].sum())
print("\n\U0001F4C5 Monthly Sales:\n", df.groupby('MONTH')["NET_SALE_AMOUNT"].sum().sort_index())

# === Filtered for Bisconni ===
bisconni_df = df[df['DIVISION'] == 'BISCONNI']
print("\n\U0001F4C8 Yearly Sales for BISCONNI:\n", bisconni_df.groupby('YEAR')['NET_SALE_AMOUNT'].sum())
print("\n\U0001F4C9 Monthly Sales for BISCONNI:\n", bisconni_df.groupby('MONTH')['NET_SALE_AMOUNT'].sum())

# === Plot Settings ===
sns.set_style("whitegrid")
def format_ticks_as_k(ax):
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1_000_000:.1f}M' if x >= 1_000_000 else f'{x/1_000:.0f}K'))

# === Region Sales ===
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='REGION', y='NET_SALE_AMOUNT', estimator=sum, ci=None, color=COMBINED_COLOR)
plt.title('Sales by Region(Both years)')
plt.xticks(rotation=45)
format_ticks_as_k(plt.gca())
plt.tight_layout()
plt.show()

# === Division Sales ===
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='DIVISION', y='NET_SALE_AMOUNT', estimator=sum, ci=None, palette={
    'BISCONNI': BISCONNI_COLOR, 'CANDYLAND': CANDYLAND_COLOR
})
plt.title('Sales by Division(Both years)')
plt.xticks(rotation=45)
format_ticks_as_k(plt.gca())
plt.tight_layout()
plt.show()

# === Yearly Sales BISCONNI vs CANDYLAND ===
filtered_df = df[df['DIVISION'].str.upper().isin(['BISCONNI', 'CANDYLAND'])]
yearly_sales = filtered_df.groupby(['YEAR', 'DIVISION'])['NET_SALE_AMOUNT'].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=yearly_sales, x='YEAR', y='NET_SALE_AMOUNT', hue='DIVISION', palette={
    'BISCONNI': BISCONNI_COLOR, 'CANDYLAND': CANDYLAND_COLOR
})
plt.title("\U0001F4C8 Yearly Sales: BISCONNI vs CANDYLAND")
plt.ylabel("Net Sales")
format_ticks_as_k(plt.gca())
plt.tight_layout()
plt.show()

# === Monthly Sales by Year (All Brands) ===
overall_monthly = df.groupby(['YEAR', 'MONTH'])['NET_SALE_AMOUNT'].sum().reset_index()
for year in sorted(overall_monthly['YEAR'].unique()):
    plt.figure(figsize=(10, 5))
    data = overall_monthly[overall_monthly['YEAR'] == year]
    sns.barplot(data=data, x='MONTH', y='NET_SALE_AMOUNT', color=COMBINED_COLOR)
    plt.title(f"\U0001F4C9 Monthly Sales (Combined) in {year}")
    plt.xticks(ticks=range(0, 12), labels=calendar.month_abbr[1:13])
    format_ticks_as_k(plt.gca())
    plt.tight_layout()
    plt.show()

# === Monthly Sales BISCONNI vs CANDYLAND per Year ===
monthly_sales = filtered_df.groupby(['YEAR', 'MONTH', 'DIVISION'])['NET_SALE_AMOUNT'].sum().reset_index()
for year in sorted(monthly_sales['YEAR'].unique()):
    plt.figure(figsize=(10, 5))
    data = monthly_sales[monthly_sales['YEAR'] == year]
    sns.barplot(data=data, x='MONTH', y='NET_SALE_AMOUNT', hue='DIVISION', palette={
        'BISCONNI': BISCONNI_COLOR, 'CANDYLAND': CANDYLAND_COLOR
    })
    plt.title(f"\U0001F4C9 Monthly Sales: BISCONNI vs CANDYLAND ({year})")
    plt.xticks(ticks=range(0, 12), labels=calendar.month_abbr[1:13])
    format_ticks_as_k(plt.gca())
    plt.tight_layout()
    plt.show()

# === Heatmaps per Division ===
heatmap_data = filtered_df.groupby(['DIVISION', 'YEAR', 'MONTH'])['NET_SALE_AMOUNT'].sum().reset_index()
for division in heatmap_data['DIVISION'].unique():
    pivot = heatmap_data[heatmap_data['DIVISION'] == division].pivot(index='YEAR', columns='MONTH', values='NET_SALE_AMOUNT').fillna(0)
    pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]
    cmap = 'Blues' if division == 'BISCONNI' else 'Oranges'
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt="", cmap=cmap,
                annot_kws={"size": 10},
                cbar_kws={'format': ticker.FuncFormatter(lambda x, _: f'{x/1_000_000:.1f}M' if x >= 1_000_000 else f'{x/1_000:.0f}K')})
    plt.title(f"\U0001F321 Monthly Sales Heatmap - {division}")
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.tight_layout()
    plt.show()