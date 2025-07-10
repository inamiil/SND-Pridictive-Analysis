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
SELECT TOP 100000000
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


#Converting function
def convert_to_millions(series):
    return series.astype(float).fillna(0) / 1e6


#For Monthly breakdowns
def group_by_year_month(df, value_col='NET_SALE_AMOUNT'):
    results = {}
    years = df['YEAR'].unique()

    for year in sorted(years):
        yearly_df = df[df['YEAR'] == year]
        grouped = (
            yearly_df.groupby('MONTH_NAME')[value_col]
            .sum()
            .reindex(month_order)
            .reset_index()
        )
        grouped[value_col] = grouped[value_col].fillna(0).astype(float) / 1e6
        results[year] = grouped

    return results  # Dictionary: {2024: DataFrame, 2025: DataFrame}


#Pie chart formatting function
def format_autopct(pct, all_vals):
    all_vals = [float(val) for val in all_vals]  # Ensure numeric
    total = sum(all_vals)
    val = pct * total / 100.0  # Convert percentage back to actual value
    return f'{pct:.1f}%\n({val/1e6:.1f}M)'  # Format as percentage and value in millions


# Group by division and sum Net Sale Amounts
division_sales = df.groupby('DIVISION')['NET_SALE_AMOUNT'].sum().reset_index()

# Clean and prepare values
division_sales = division_sales[division_sales['NET_SALE_AMOUNT'] > 0].dropna()
labels = division_sales['DIVISION']
sizes = division_sales['NET_SALE_AMOUNT'].astype(float)
colors = ['#66b3ff', '#ff9999']  # Optional custom colors

# Draw Pie Chart
def plot_pie_chart_by_category(df, category_col, title=None, colors=None):
    """
    Plots a pie chart for a given category column using NET_SALE_AMOUNT.
    
    Parameters:
    - df: DataFrame
    - category_col: str, e.g., 'DIVISION', 'REGION', 'AREA'
    - title: str, optional title for the chart
    - colors: list of colors for the pie chart
    """

    #Grouping & Cleaning
    grouped = df.groupby(category_col)['NET_SALE_AMOUNT'].sum().reset_index()
    grouped = grouped[grouped['NET_SALE_AMOUNT'] > 0].dropna()

    labels = grouped[category_col]
    sizes = grouped['NET_SALE_AMOUNT'].astype(float)
    
    if not colors:
        colors = sns.color_palette("Set2", len(labels))

    # Step 2: Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})

    # Pie chart on the left
    wedges, texts, autotexts = ax1.pie(
    sizes,
    labels=labels,
    autopct=lambda pct: format_autopct(pct, sizes),
    startangle=140,
    colors=colors,
    textprops={'fontsize': 10}
    )
    ax1.axis('equal')
    ax1.set_title(title or f'Net Sale Amount by {category_col.capitalize()}', fontsize=14)

    # Value labels on the right
    lines = [f"{label}: {val / 1e6:.1f}M" for label, val in zip(labels, sizes)]
    text_str = "\n".join(lines)
    ax2.axis('off')
    ax2.text(0, 1, text_str, ha='left', va='top', fontsize=12)

    plt.tight_layout()
    plt.show()

#Draw Single Entity Bar Chart for Total Sales
def plot_bar_chart_by_category(df, category_col, title=None, colors=None):
    """
    Plots a bar chart for NET_SALE_AMOUNT grouped by a single category (e.g., DIVISION, REGION, AREA).
    
    Parameters:
    - df: DataFrame
    - category_col: str, column name to group by
    - title: str, optional chart title
    - colors: list of colors, optional
    """
    # Group and clean
    grouped = df.groupby(category_col)['NET_SALE_AMOUNT'].sum().reset_index()
    grouped = grouped[grouped['NET_SALE_AMOUNT'] > 0].dropna()

    labels = grouped[category_col]
    sizes = grouped['NET_SALE_AMOUNT'].astype(float) / 1e6  # Convert to millions

    if not colors:
        colors = sns.color_palette("Set2", len(labels))

    # Plotting
    fig_width = max(8, len(labels) * 0.75)  # Adjust width: 0.75 inches per category, min 8 inches
    plt.figure(figsize=(fig_width, 6))
    bars = plt.bar(labels, sizes, color=colors)

    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.1f}M', ha='center', fontsize=12)

    # Formatting
    if not title:
        title = f'Net Sale Amount by {category_col.capitalize()}'
    plt.title(title, fontsize=14)
    plt.xlabel(category_col.capitalize())
    plt.ylabel('Net Sale Amount (Millions)', fontsize=12)
    plt.tight_layout()
    plt.show()

#Segregating and grouping
def plot_grouped_bar_chart(df, group_cols, value_col='NET_SALE_AMOUNT', title='', ylabel='', color_map=None):
    """
    Plots a grouped bar chart for two categories (e.g., REGION vs DIVISION).

    Parameters:
    - df: DataFrame
    - group_cols: List of two column names to group by, e.g. ['REGION', 'DIVISION']
    - value_col: Column to sum and plot (default: NET_SALE_AMOUNT)
    - title: Chart title
    - ylabel: Y-axis label
    - color_map: Optional dict mapping category (e.g. divisions) to color
    """

    # Pivot the data
    grouped = df.groupby(group_cols)[value_col].sum().unstack().reset_index()

    # Convert values to millions
    for col in grouped.columns[1:]:
        grouped[col] = grouped[col].astype(float) / 1e6

    # Set bar widths and positions
    x_labels = grouped[group_cols[0]]
    categories = grouped.columns[1:]
    x = np.arange(len(x_labels))
    bar_width = 0.8 / len(categories)  # Adjust width per number of bars per group

    # Plot
    plt.figure(figsize=(max(10, len(x_labels) * 1.2), 6))

    for i, cat in enumerate(categories):
        color = color_map[cat] if color_map and cat in color_map else None
        plt.bar(
            x + (i - len(categories)/2) * bar_width + bar_width/2,
            grouped[cat],
            width=bar_width,
            label=cat,
            color=color
        )
        # Add value labels
        for j in range(len(x)):
            val = grouped[cat].iloc[j]
            plt.text(x[j] + (i - len(categories)/2) * bar_width + bar_width/2, val + 0.1, f'{val:.1f}M', ha='center', fontsize=9)

    plt.xlabel(group_cols[0])
    plt.ylabel(ylabel if ylabel else 'Net Sale Amount (Millions)')
    plt.title(title if title else f'{group_cols[1]}-wise Sales by {group_cols[0]}')
    plt.xticks(x, x_labels)
    plt.legend(title=group_cols[1])
    plt.tight_layout()
    plt.show()

#Line Graph for single entities for Total Sales 
def plot_monthly_line_by_entity_year(df, entity_col, title_prefix="Monthly Net Sales Trend", color='#4c72b0'):
    """
    Plots a monthly line chart for each unique entity (e.g., Division, Region, Area) per year.
    
    Parameters:
    - df: DataFrame
    - entity_col: str, e.g. 'DIVISION', 'REGION', 'AREA'
    - title_prefix: str, prefix for the chart title
    - color: color for the line
    """
    # Ensure MONTH_NAME is present
    if 'MONTH_NAME' not in df.columns:
        df['MONTH_NAME'] = df['MONTH'].apply(lambda x: calendar.month_abbr[x])

    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    entities = df[entity_col].unique()
    years = df['YEAR'].unique()

    for entity in entities:
        for year in years:
            filtered = df[(df[entity_col] == entity) & (df['YEAR'] == year)]

            grouped = (
                filtered.groupby('MONTH_NAME')['NET_SALE_AMOUNT']
                .sum()
                .reindex(month_order)
                .reset_index()
            )

            grouped['NET_SALE_AMOUNT'] = grouped['NET_SALE_AMOUNT'].fillna(0).astype(float) / 1e6

            # Plot
            plt.figure(figsize=(10, 5))
            plt.plot(
                grouped['MONTH_NAME'],
                grouped['NET_SALE_AMOUNT'],
                marker='o',
                label=entity,
                color=color
            )

            for i in range(len(grouped)):
                y = grouped['NET_SALE_AMOUNT'][i]
                plt.text(i, y + 0.1, f"{y:.1f}M", ha='center', fontsize=10)

            plt.title(f'{title_prefix} – {entity} ({year})')
            plt.xlabel('Month')
            plt.ylabel('Net Sale Amount (Millions)')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

#Top 5 of any Category for Total Sales
def plot_top5_bar_by_category(df, category_col, title_prefix="Top 5", color='#66b3ff'):
    # Step 1: Group and sort
    grouped = df.groupby(category_col)['NET_SALE_AMOUNT'].sum().reset_index()
    top5 = grouped.sort_values(by='NET_SALE_AMOUNT', ascending=False).head(5)

    # Step 2: Prepare for bar plot
    labels = top5[category_col]
    values = top5['NET_SALE_AMOUNT'].astype(float) / 1e6

    # Step 3: Plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=color)

    # Step 4: Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.1f}M', ha='center', fontsize=12)

    # Step 5: Title and labels
    plt.title(f"{title_prefix} {category_col.title()} by Sales (Millions)", fontsize=14)
    plt.xlabel(category_col.title())
    plt.ylabel("Net Sale Amount (Millions)")
    plt.tight_layout()
    plt.show()


#Top 5 Line graph for Monthly Trends
def plot_top5_monthly_trend(df, category_col, title_prefix='Top 5 Monthly Trend'):
    """
    Plots monthly line graphs for top 5 categories (by total sales), one per year.

    Parameters:
    - df: DataFrame with 'MONTH_NAME', 'YEAR', and 'NET_SALE_AMOUNT'
    - category_col: e.g., 'AREA', 'REGION', 'DIVISION'
    - title_prefix: Base title for the graph
    """

    # Get top 5 based on overall NET_SALE_AMOUNT
    df[category_col] = df[category_col].astype(str)  # Ensure it's a string (just in case)

    # ✅ Convert NET_SALE_AMOUNT to float before nlargest
    df['NET_SALE_AMOUNT'] = pd.to_numeric(df['NET_SALE_AMOUNT'], errors='coerce')

    top5 = (
        df.groupby(category_col)['NET_SALE_AMOUNT']
        .sum()
        .dropna()
        .nlargest(5)
        .index
    )

    # Filter the dataframe to only include top 5
    df_top5 = df[df[category_col].isin(top5)].copy()

    # Get years
    years = df_top5['YEAR'].unique()

    # Plot for each year
    for year in years:
        yearly_df = df_top5[df_top5['YEAR'] == year]

        plt.figure(figsize=(10, 6))
        for cat in top5:
            line_df = yearly_df[yearly_df[category_col] == cat]
            grouped = (
                line_df.groupby('MONTH_NAME')['NET_SALE_AMOUNT']
                .sum()
                .reindex(month_order)
                .fillna(0)
                .reset_index()
            )
            grouped['NET_SALE_AMOUNT'] = grouped['NET_SALE_AMOUNT'].astype(float) / 1e6

            plt.plot(
                grouped['MONTH_NAME'],
                grouped['NET_SALE_AMOUNT'],
                marker='o',
                label=cat
            )

        plt.title(f'{title_prefix} – {year}')
        plt.xlabel('Month')
        plt.ylabel('Net Sale Amount (Millions)')
        plt.xticks(rotation=45)
        plt.legend(title=category_col)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

#---DIVISION---#


#DIVISION - PIE CHART
plot_pie_chart_by_category(
    df,
    category_col='DIVISION',
    title='Total Sales per Division (Jan 2024 – April 2025)',
    colors=['#66b3ff', '#ff9999']
)

#DIVISION - BAR GRAPH (TOTAL SALES)
plot_bar_chart_by_category(
    df,
    category_col='DIVISION',
    title='Total Sales by Division (Jan 2024 – Apr 2025)',
    colors=['#66b3ff', '#ff9999']
)

#DIVISION - LINE GRAPH (MONTHLY SALES)
plot_monthly_line_by_entity_year(df, 'DIVISION', title_prefix="Monthly Net Sales Trend Per Division", color='#4c72b0')


#---REGION---#


#REGION - PIE CHART
plot_pie_chart_by_category(
    df,
    category_col='REGION',
    title='Total Sales by Region (Jan 2024 – April 2025)',
    colors=['#ffcc99', '#99ccff']
)

#REGION - BAR GRAPH (TOTAL SALES)
plot_bar_chart_by_category(
    df,
    category_col='REGION',
    title='Total Sales by REGION (Jan 2024 – Apr 2025)',
    colors=['#66b3ff', '#ff9999']
)


#REGION & DIVISION - BAR GRAPH (TOTAL SALES)
plot_grouped_bar_chart(
    df,
    group_cols=['REGION', 'DIVISION'],
    title='Division-wise Net Sales by Region',
    ylabel='Net Sale Amount (Millions)',
    color_map={'CANDYLAND': '#66b3ff', 'BISCONNI': '#ff9999'}
)


#REGION - LINE GRAPH (MONTHLY SALES)
plot_monthly_line_by_entity_year(df, 'REGION', title_prefix="Monthly Net Sales Trend BY REGION", color='#4c72b0')


#---AREA---#


#AREA - PIE CHART
plot_pie_chart_by_category(
    df,
    category_col='AREA',
    title='Total Sales by Area (Jan 2024 – April 2025)'
)


#Area - BAR GRAPH (Total Sales)
plot_bar_chart_by_category(
    df,
    category_col='AREA',
    title='Total Sales by AREA (Jan 2024 – Apr 2025)',
)

#AREA & REGION - BAR GRAPH (TOTAL SALES)
plot_grouped_bar_chart(
    df,
    group_cols=['REGION', 'AREA'],
    title='Area-wise Net Sales by Region',
    ylabel='Net Sale Amount (Millions)'
)


#AREA - TOP 5 AREAS (TOTAL SALES)
plot_top5_bar_by_category(df, 'AREA', title_prefix="Top 5 Area-wise Total Sales", color='#4c72b0')


#AREA - TOP 5 (MONTHLY TREND)
plot_top5_monthly_trend(df, category_col='AREA')


#AREA - LINE GRAPH (MONTHLY SALES)
#plot_monthly_line_by_entity_year(df, 'AREA', title_prefix="Monthly Net Sales Trend", color='#4c72b0')


#---ZONE---#


#ZONE - PIE CHART
#plot_pie_chart_by_category(df,category_col='ZONE',title='Total Sales by Zone (Jan 2024 – April 2025)')


#ZONE - BAR GRAPH (Total Sales)
#plot_bar_chart_by_category(df,category_col='ZONE',title='Total Sales by ZONE (Jan 2024 – Apr 2025)',)

#ZONE & AREA - BAR GRAPH (TOTAL SALES)
#plot_grouped_bar_chart(df,group_cols=['AREA', 'ZONE'],title='Zone-wise Net Sales by Area',ylabel='Net Sale Amount (Millions)')


#ZONE- TOP 5 ZONES (TOTAL SALES)
plot_top5_bar_by_category(df, 'ZONE', title_prefix="Top 5 Zone-wise Total Sales", color='#4c72b0')


#ZONE - TOP 5 (MONTHLY TREND)
plot_top5_monthly_trend(df, category_col='ZONE')


#ZONE - LINE GRAPH (MONTHLY SALES)
#plot_monthly_line_by_entity_year(df, 'ZONE', title_prefix="Monthly Net Sales Trend", color='#4c72b0')


#---TERRITORY---#


#TERRITORY - PIE CHART
#plot_pie_chart_by_category(df,category_col='TERRITORY',title='Total Sales by Territory (Jan 2024 – April 2025)')


#TERRITORY - BAR GRAPH (Total Sales)
#plot_bar_chart_by_category(df,category_col='TERRITORY',title='Total Sales by TERRITORY (Jan 2024 – Apr 2025)',)

#TERRITORY & ZONE - BAR GRAPH (TOTAL SALES)
#plot_grouped_bar_chart(df,group_cols=['ZONE', 'TERRITORY'],title='Territory-wise Net Sales by ZONE',ylabel='Net Sale Amount (Millions)')


#TERRITORY - TOP 5 TERRITORIES (TOTAL SALES)
plot_top5_bar_by_category(df, 'TERRITORY', title_prefix="Top 5 Territory-wise Total Sales", color='#4c72b0')


#TERRITORIES - TOP 5 (MONTHLY TREND)
plot_top5_monthly_trend(df, category_col='TERRITORY')


#TERRITORY - LINE GRAPH (MONTHLY SALES)
#plot_monthly_line_by_entity_year(df, 'TERRITORY', title_prefix="Monthly Net Sales Trend", color='#4c72b0')


#---TOWN---#


#TOWN - PIE CHART
#plot_pie_chart_by_category(df,category_col='TOWN',title='Total Sales by TOWN (Jan 2024 – April 2025)')


#TOWN - BAR GRAPH (Total Sales)
#plot_bar_chart_by_category(df,category_col='TOWN',title='Total Sales by TOWN (Jan 2024 – Apr 2025)',)

#TOWN & TERRITORY - BAR GRAPH (TOTAL SALES)
#plot_grouped_bar_chart(df,group_cols=['TERRITORY', 'TOWN'],title='Town-wise Net Sales by Territory',ylabel='Net Sale Amount (Millions)')


#TOWN- TOP 5 TOWN (TOTAL SALES)
plot_top5_bar_by_category(df, 'TOWN', title_prefix="Top 5 Town-wise Total Sales", color='#4c72b0')


#TOWN - TOP 5 (MONTHLY TREND)
plot_top5_monthly_trend(df, category_col='TOWN')


#TOWN - LINE GRAPH (MONTHLY SALES)
#plot_monthly_line_by_entity_year(df, 'TOWN', title_prefix="Monthly Net Sales Trend", color='#4c72b0')

