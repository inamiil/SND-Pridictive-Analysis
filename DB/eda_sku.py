import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from db_connect import get_connection
from db_connect import run_query

# === Load Data ===
sku_query = """SELECT SKU_CODE, SKU_LDESC, NET_SALE_AMOUNT, DIVISION, PACK_SIZE, FLAVOUR, BRAND FROM FINAL_QUERY """

data = run_query(sku_query)
df = pd.DataFrame(data)
df.head()

# Optional: ensure numeric column type
df['NET_SALE_AMOUNT'] = pd.to_numeric(df['NET_SALE_AMOUNT'], errors='coerce')

# === Set Style ===
sns.set_style("whitegrid")

# === Formatter Function ===
def format_ticks_as_k(ax):
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f'{x/1_000_000:.1f}M' if x >= 1_000_000 else f'{x/1_000:.0f}K')
    )

# ---- 1. TOP 10 SELLING SKUs (Combined) ----
top_skus = (
    df.groupby(['SKU_CODE', 'SKU_LDESC'])['NET_SALE_AMOUNT'].sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

print("\n🎯 Top 10 Selling SKUs (Combined):")
print(top_skus)

plt.figure(figsize=(10, 5))
sns.barplot(data=top_skus, y='SKU_LDESC', x='NET_SALE_AMOUNT', palette='viridis')
plt.title("Top 10 SKUs by Net Sale Amount (Combined)")
plt.xlabel("Net Sale Amount")
plt.ylabel("SKU Description")
format_ticks_as_k(plt.gca())
plt.tight_layout()
plt.show()

# ---- 2. SALES DISTRIBUTION ACROSS PACK SIZES ----
for division in ['CANDYLAND', 'BISCONNI']:
    pack_sales = (
        df[df['DIVISION'] == division]
        .groupby('PACK_SIZE')['NET_SALE_AMOUNT']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    print(f"\n📦 Sales by Pack Size in {division}:")
    print(pack_sales)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=pack_sales, x='PACK_SIZE', y='NET_SALE_AMOUNT', palette='cubehelix')
    plt.title(f"Sales Distribution by Pack Size - {division}")
    plt.xticks(rotation=45)
    format_ticks_as_k(plt.gca())
    plt.tight_layout()
    plt.show()

# ---- 3. TOP BRANDS (Separated by Division) ----
for division in ['CANDYLAND', 'BISCONNI']:
    top_brands = (
        df[df['DIVISION'] == division]
        .groupby('BRAND')['NET_SALE_AMOUNT']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    print(f"\n🌟 Top Brands in {division}:")
    print(top_brands)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_brands, x='BRAND', y='NET_SALE_AMOUNT', palette='crest')
    plt.title(f"Top 10 Brands - {division}")
    plt.xticks(rotation=45)
    format_ticks_as_k(plt.gca())
    plt.tight_layout()
    plt.show()

# ---- 4. BRAND PERFORMANCE BY PACK SIZE ----
for division in ['CANDYLAND', 'BISCONNI']:
    brand_pack_sales = (
        df[df['DIVISION'] == division]
        .groupby(['BRAND', 'PACK_SIZE'])['NET_SALE_AMOUNT']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    print(f"\n📦 Top Brand-PackSize Combos in {division}:")
    print(brand_pack_sales)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=brand_pack_sales, x='PACK_SIZE', y='NET_SALE_AMOUNT', hue='BRAND')
    plt.title(f"Top 10 Brand-Pack Size Sales - {division}")
    plt.xticks(rotation=45)
    format_ticks_as_k(plt.gca())
    plt.tight_layout()
    plt.show()

# ---- 5. TOP FLAVOURS BY DIVISION ----
for division in ['CANDYLAND', 'BISCONNI']:
    top_flavours = (
        df[df['DIVISION'] == division]
        .groupby('FLAVOUR')['NET_SALE_AMOUNT']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    print(f"\n🍧 Top Flavours in {division}:")
    print(top_flavours)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_flavours, x='FLAVOUR', y='NET_SALE_AMOUNT', palette='magma')
    plt.title(f"Top 10 Flavours - {division}")
    plt.xticks(rotation=45)
    format_ticks_as_k(plt.gca())
    plt.tight_layout()
    plt.show()
