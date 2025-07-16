import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from db_connect import run_query

# === SQL Query: Top Distributors Overall ===
query = """
SELECT 
    DISTRIBUTOR_NAME,
    DIVISION,
    SUM(NET_SALE_AMOUNT) AS TOTAL_SALES
FROM FINAL_QUERY
WHERE DIVISION IN ('BISCONNI', 'CANDYLAND')
GROUP BY DISTRIBUTOR_NAME, DIVISION
"""

# === Load Data ===
df = pd.DataFrame(run_query(query))

# === Get Top 5 Distributors Overall ===
top5 = (
    df.groupby("DISTRIBUTOR_NAME")["TOTAL_SALES"]
    .sum()
    .nlargest(5)
    .reset_index()
)

# === Merge to Get Division Info ===
df_top5 = df[df["DISTRIBUTOR_NAME"].isin(top5["DISTRIBUTOR_NAME"])]

# === Format Sales Numbers ===
df_top5["SALES_LABEL"] = df_top5["TOTAL_SALES"].apply(
    lambda x: f"{x/1_000_000:.1f}M" if x < 1_000_000_000 else f"{x/1_000_000_000:.2f}B"
)

# === Print Numeric Summary ===
print("\n===============================")
print("TOP 5 DISTRIBUTORS - TOTAL SALES")
print("===============================")
print(df_top5[["DISTRIBUTOR_NAME", "DIVISION", "TOTAL_SALES", "SALES_LABEL"]])

# === Bar Chart ===
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_top5,
    x="DISTRIBUTOR_NAME",
    y="TOTAL_SALES",
    hue="DIVISION",
    palette={"BISCONNI": "orange", "CANDYLAND": "blue"}
)
plt.title("Top 5 Distributors by Total Sales")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Pie Chart ===
plt.figure(figsize=(7, 7))
plt.pie(
    top5["TOTAL_SALES"],
    labels=top5["DISTRIBUTOR_NAME"],
    autopct="%1.1f%%",
    startangle=140
)
plt.title("Sales Share of Top 5 Distributors")
plt.tight_layout()
plt.show()
