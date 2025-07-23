import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from db_connect import get_top_cocomo_skus  # <-- Make sure this function exists

def billions_formatter(x, pos):
    """
    Custom formatter to convert large numbers to readable strings.
    """
    if x >= 1e9:
        return f'{x * 1e-9:.1f}B'
    elif x >= 1e6:
        return f'{x * 1e-6:.1f}M'
    elif x >= 1e3:
        return f'{x * 1e-3:.0f}K'
    else:
        return f'{x:.0f}'

def plot_cocomo_sku_monthly_trend():
    df = get_top_cocomo_skus()  # Fetch COCOMO data

    # Create a datetime column for time-based plotting
    df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
    df = df.sort_values(['SKU_CODE', 'DATE'])

    plt.figure(figsize=(14, 7))
    ax = sns.lineplot(
        data=df,
        x='DATE',
        y='MONTHLY_SALES',
        hue='SKU_LDESC',
        marker='o',
        palette='tab10'
    )

    # Chart labels and formatting
    plt.title('Monthly Sales Trend of Top 5 COCOMO SKUs', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Net Sales Amount', fontsize=12)

    # Format Y-axis
    ax.yaxis.set_major_formatter(FuncFormatter(billions_formatter))

    # Format X-axis
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)
    plt.legend(title='SKU Description')
    plt.show()

# Run the COCOMO plot
plot_cocomo_sku_monthly_trend()
