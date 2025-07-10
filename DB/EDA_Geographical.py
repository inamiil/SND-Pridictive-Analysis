#Libaries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from db_connect import get_connection 
from db_connect import run_query
import numpy as np

#connecting to database
conn = get_connection()

#Querie for Geographical View
geo_query = """
SELECT TOP 100000
    DATE_ID,
    DAY,
    MONTH,
    YEAR,
    DIVISION,
    REGION,
    AREA,
    ZONE,
    TERRITORY,
    TOWN,
    NET_SALES_UNITS,
    NET_SALES_GROSS_VALUE,
    NET_SALE_AMOUNT,
    NET_DISCOUNT_VALUE
FROM FINAL_QUERY
    """

#loading into dataframe
data = run_query(geo_query)

#Converting to dataframe
df = pd.DataFrame(data)

# Ensure MONTH is numeric
df['MONTH'] = df['MONTH'].astype(int)

# Add short month names (e.g., Jan, Feb, ...)
df['MONTH_NAME'] = df['MONTH'].apply(lambda x: calendar.month_abbr[x])

df = df.sort_values(by=['YEAR', 'MONTH'])

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
# Set style
sns.set(style="whitegrid")


# Formatting
def format_autopct(pct, all_vals):
    all_vals = [float(val) for val in all_vals]  # ✅ Convert to float
    total = sum(all_vals)
    val = pct * total / 100.0
    return f'{pct:.1f}%\n({val/1e6:.1f}M)'

#---DIVISION---

#Sales per division per region
region_div_sales = df.groupby(['REGION', 'DIVISION'])['NET_SALE_AMOUNT'].sum().reset_index()

#sales per REGION (regardless of division)
region_total_sales = df.groupby('REGION')['NET_SALE_AMOUNT'].sum().reset_index()
region_total_sales['DIVISION'] = 'Total'  # To match column names for merging later

# Combine both into one DataFrame
combined = pd.concat([region_div_sales, region_total_sales])

# Pie Chart
labels = ['BISCONNI', 'CANDYLAND']  # Placeholder labels, replace later
div_sizes = df['NET_SALE_AMOUNT']
colors = ['#66b3ff', '#ff9999']   # Optional custom colors

# Group the data by DIVISION (do this first!)
division_sales = df.groupby('DIVISION')['NET_SALE_AMOUNT'].sum().reset_index()

# Remove any negative or NaN values (safety step)
division_sales = division_sales[division_sales['NET_SALE_AMOUNT'] > 0].dropna()

# Prepare data for plotting
labels = division_sales['DIVISION']
div_sizes = division_sales['NET_SALE_AMOUNT'].astype(float)
div_sizes = div_sizes.fillna(0)         # Replace NaN with 0
div_sizes = div_sizes.clip(lower=0)     # Remove negative values

colors = ['#66b3ff', '#ff9999']

# Drawing PieChart - DIVISION
plt.figure(figsize=(8, 8))
plt.pie(
    div_sizes,
    labels=labels,
    autopct=lambda pct: format_autopct(pct, div_sizes),  # 💡 Custom value formatting here
    startangle=140,
    colors=colors,
    textprops={'fontsize': 12}
)
plt.title('Net Sale Amount Distribution by Division (Jan 2024 - May 2025)')
plt.axis('equal')  # Makes the pie chart circular

# Values under Plot
value_lines = "\n".join([f"{label}: {amount/1e6:.1f}M" for label, amount in zip(labels, div_sizes)])
plt.text(0, -1.3, value_lines, ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.show()

# Bar chart - DIVISION
plt.figure(figsize=(8, 6))
bars = plt.bar(
    labels,
    div_sizes / 1e6,  # Convert to millions
    color=colors
)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.1f}M', ha='center', fontsize=12)

# Chart formatting
plt.title('Net Sale Amount by Division', fontsize=14)
plt.xlabel('Division')
plt.ylabel('Net Sale Amount (Millions)', fontsize=12)
plt.tight_layout()
plt.show()

#Line Graph - DIVISION

#Add month short names
df['MONTH_NAME'] = df['MONTH'].apply(lambda x: calendar.month_abbr[x])

# Define correct month order
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Now plot one graph per division per year
divisions = df['DIVISION'].unique()
years = df['YEAR'].unique()

for division in divisions:
    for year in years:
        div_year_df = df[(df['DIVISION'] == division) & (df['YEAR'] == year)]

        # Group and sort
        grouped = div_year_df.groupby('MONTH_NAME')['NET_SALE_AMOUNT'].sum().reindex(month_order).reset_index()
        grouped['NET_SALE_AMOUNT'] = grouped['NET_SALE_AMOUNT'].fillna(0)
        grouped['NET_SALE_AMOUNT'] = grouped['NET_SALE_AMOUNT'].apply(float)
        grouped['NET_SALE_AMOUNT'] = grouped['NET_SALE_AMOUNT'] / 1e6

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(grouped['MONTH_NAME'], grouped['NET_SALE_AMOUNT'].astype(float) / 1e6, marker='o', label=division, color='#4c72b0')
        plt.title(f'Monthly Net Sales Trend – {division} ({year})')
        plt.xlabel('Month')
        plt.ylabel('Net Sale Amount (Millions)')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

#--- REGION ---

# Group the data by REGION
region_sales = df.groupby('REGION')['NET_SALE_AMOUNT'].sum().reset_index()

# Optional safety check: remove any NaNs or negative values
region_sales = region_sales[region_sales['NET_SALE_AMOUNT'] > 0].dropna()

# Prepare labels and values
labels = region_sales['REGION'].tolist()
reg_sizes = region_sales['NET_SALE_AMOUNT'].astype(float)
reg_sizes = reg_sizes.fillna(0)         # Replace NaN with 0
reg_sizes = reg_sizes.clip(lower=0)     # Remove negative values

colors = ['#ffcc99', '#99ccff']  # Optional colors

#Pie Chart - REGION
plt.figure(figsize=(8, 8))
plt.pie(
    reg_sizes,
    labels=labels,
    autopct=lambda pct: format_autopct(pct, reg_sizes),
    startangle=140,
    colors=colors,
    textprops={'fontsize': 12}
)
plt.title('Net Sale Amount Distribution by Region (Jan 2024 - May 2025)', fontsize=14)
plt.axis('equal')  # Ensures the pie chart is circular

# Add values under the chart
value_lines = "\n".join([f"{label}: {amount/1e6:.1f}M" for label, amount in zip(labels, reg_sizes)])
plt.text(0, -1.3, value_lines, ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.show()

# Assuming region_sales already exists:
region_sales = df.groupby('REGION')['NET_SALE_AMOUNT'].sum().reset_index()

# Bar chart - REGION
plt.figure(figsize=(8, 6))
bars = plt.bar(
    region_sales['REGION'],
    reg_sizes / 1e6,  # Convert to millions
    color=['#ffcc99', '#99ccff']
)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.1f}M', ha='center', fontsize=12)

# Chart formatting
plt.title('Net Sale Amount by Region', fontsize=14)
plt.xlabel('Region')
plt.ylabel('Net Sale Amount (Millions)', fontsize=12)
plt.tight_layout()
plt.show()

#Bar Chart - REGION and DIVISION
grouped = df.groupby(['REGION', 'DIVISION'])['NET_SALE_AMOUNT'].sum().unstack().reset_index()

# Convert to millions
grouped[['BISCONNI', 'CANDYLAND']] = grouped[['BISCONNI', 'CANDYLAND']].astype(float) 
grouped[['BISCONNI', 'CANDYLAND']] = grouped[['BISCONNI', 'CANDYLAND']] / 1e6

# Setup for grouped bars
regions = grouped['REGION']
bar_width = 0.35
x = np.arange(len(regions))  # X-axis positions

# Create plot
plt.figure(figsize=(10, 6))
plt.bar(x - bar_width/2, grouped['CANDYLAND'], width=bar_width, label='CANDYLAND', color='#66b3ff')
plt.bar(x + bar_width/2, grouped['BISCONNI'], width=bar_width, label='BISCONNI', color='#ff9999')

# Add labels on top
for i in range(len(regions)):
    plt.text(x[i] - bar_width/2, grouped['CANDYLAND'][i] + 0.2, f"{grouped['CANDYLAND'][i]:.1f}M", ha='center', fontsize=10)
    plt.text(x[i] + bar_width/2, grouped['BISCONNI'][i] + 0.2, f"{grouped['BISCONNI'][i]:.1f}M", ha='center', fontsize=10)

# Formatting
plt.xlabel('Region')
plt.ylabel('Net Sale Amount (Millions)')
plt.title('Division-wise Net Sales by Region')
plt.xticks(x, regions)
plt.legend()
plt.tight_layout()
plt.show()

#Line Chart - REGION AND DIVISION monthly
regions = df['REGION'].unique()
years = df['YEAR'].unique()

for region in regions:
    for year in years:
        reg_year_df = df[(df['REGION'] == region) & (df['YEAR'] == year)]

        # Group and sort by month name using month_order
        grouped = reg_year_df.groupby('MONTH_NAME')['NET_SALE_AMOUNT'].sum().reindex(month_order).reset_index()
        grouped['NET_SALE_AMOUNT'] = grouped['NET_SALE_AMOUNT'].fillna(0)
        grouped['NET_SALE_AMOUNT'] = grouped['NET_SALE_AMOUNT'].astype(float)
        grouped['NET_SALE_AMOUNT'] = grouped['NET_SALE_AMOUNT'] / 1e6  # Convert to millions

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(
            grouped['MONTH_NAME'],
            grouped['NET_SALE_AMOUNT'].astype(float) / 1e6,
            marker='o',
            label=region,
            color='#dd8452'
        )

        for i in range(len(grouped)):
            plt.text(i, grouped['NET_SALE_AMOUNT'][i] + 0.1, f"{grouped['NET_SALE_AMOUNT'][i]:.1f}M", ha='center', fontsize=10)

        plt.title(f'Monthly Net Sales Trend – {region} ({year})')
        plt.xlabel('Month')
        plt.ylabel('Net Sale Amount (Millions)')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

#Bar Chart - Division and Region monthly
divisions = df['DIVISION'].unique()
regions = df['REGION'].unique()

for division in divisions:
    for region in regions:
        # Filter data
        filtered = df[(df['DIVISION'] == division) & (df['REGION'] == region)].copy()
        if filtered.empty:
            continue

        # Group by YEAR and MONTH
        grouped = (
            filtered.groupby(['YEAR', 'MONTH'])['NET_SALE_AMOUNT']
            .sum()
            .reset_index()
        )
        grouped['NET_SALE_AMOUNT'] = grouped['NET_SALE_AMOUNT'].astype(float) 
        grouped['NET_SALE_AMOUNT'] = grouped['NET_SALE_AMOUNT']/ 1e6  # Convert to millions
        grouped['MONTH_NAME'] = grouped['MONTH'].apply(lambda x: calendar.month_abbr[x])

        # Plot separately for each year
        for year in [2024, 2025]:
            year_data = grouped[grouped['YEAR'] == year].sort_values('MONTH')

            if year_data.empty:
                continue

            plt.figure(figsize=(10, 5))
            bars = plt.bar(
                year_data['MONTH_NAME'],
                year_data['NET_SALE_AMOUNT'],
                color='#66b3ff' if division.upper() == 'CANDYLAND' else '#ff9999'
            )

            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.1f}M', ha='center', fontsize=10)

            plt.title(f'{division} – {region} – Monthly Sales ({year})', fontsize=14)
            plt.xlabel('Month')
            plt.ylabel('Net Sale Amount (Millions)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

#--- AREA ---

#Bar Chart - Monthly of all Areas
areas = df['AREA'].unique()
for area in areas:
    area_df = df[df['AREA'] == area]

    # Group for 2024
    df_2024 = area_df[area_df['YEAR'] == 2024]
    grouped_2024 = div_year_df.groupby('MONTH_NAME')['NET_SALE_AMOUNT'].sum().reindex(month_order).reset_index()
    grouped_2024['NET_SALE_AMOUNT'] = grouped_2024['NET_SALE_AMOUNT'].astype(float)
    grouped_2024['NET_SALE_AMOUNT'] = grouped_2024['NET_SALE_AMOUNT'] / 1e6

    # Plot for 2024
    plt.figure(figsize=(10, 5))
    bars = plt.bar(grouped_2024['MONTH_NAME'], grouped_2024['NET_SALE_AMOUNT'], color='#4c72b0')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{bar.get_height():.1f}M', ha='center', fontsize=9)
    plt.title(f'{area} – Monthly Sales (2024)')
    plt.xlabel('Month')
    plt.ylabel('Net Sales (Millions)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Group for 2025
    df_2025 = area_df[area_df['YEAR'] == 2025]
    grouped_2025 = df_2025.groupby('MONTH_NAME')['NET_SALE_AMOUNT'].sum().reindex(month_order).reset_index()
    grouped_2025['NET_SALE_AMOUNT'] = grouped_2025['NET_SALE_AMOUNT'].astype(float)
    grouped_2025['NET_SALE_AMOUNT'] = grouped_2025['NET_SALE_AMOUNT'] / 1e6

    # Plot for 2025
    plt.figure(figsize=(10, 5))
    bars = plt.bar(grouped_2025['MONTH_NAME'], grouped_2025['NET_SALE_AMOUNT'], color='#4c72b0')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{bar.get_height():.1f}M', ha='center', fontsize=9)
    plt.title(f'{area} – Monthly Sales (2025)')
    plt.xlabel('Month')
    plt.ylabel('Net Sales (Millions)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#MONTHLY SORTING
#monthly_grouped = df.groupby('MONTH_NAME')['NET_SALE_AMOUNT'].sum().reindex(month_order).reset_index()
#monthly_grouped['NET_SALE_AMOUNT'] = monthly_grouped['NET_SALE_AMOUNT'].fillna(0) / 1e6 
