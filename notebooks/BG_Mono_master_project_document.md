# Project Master File

## Project Structure

```
├── .gitignore
├── .vscode
│   └── launch.json
├── README.md
├── bats
│   └── report_run_bg.bat
├── fred_fun.py
├── fred_snapshot.log
├── generate_report.log
├── generate_report.py
├── notebooks
│   ├── db_connect.ipynb
│   └── master_markdown.ipynb
├── old.py
├── reference
│   ├── constants.py
│   └── fred.xlsx
├── sql
│   └── sm_ddl.sql
├── templates
│   ├── 0_base.html
│   ├── 0_full_report.html
│   ├── 2_macro_dash.html
│   ├── 3_inflation.html
│   └── 4_gdp.html
└── utils.py
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\fred_fun.py

```py
from fredapi import Fred
from pathlib import Path
import pandas as pd
import xlwings as xw
import logging
import time

fred = Fred(api_key='37eb22bada238c97f282715480e7d897')

REFRESH_DATA = False  # Set to True to pull new data from FRED
MASTER_FILE = Path('reference') / 'fred.xlsx'
FRED_DATA_DUMP_PATH = Path('output') / 'fred_data_dump.xlsx'
DASHBOARD_1_EXPORT_PATH = Path('output') / 'fred_dashboard_1.xlsx'
SLEEP_TIME = 0.1  # seconds

# Set up basic logging config to log to console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fred_snapshot.log'),
        logging.StreamHandler()
    ]
)


def download_fred_data(fred_ids: list[str],save_to_output:bool=True) -> pd.DataFrame:
    """
    Save all of the FRED data that is downloaded into a single, long format DataFrame.
    This is useful for debugging and checking the data that is being pulled. 
    """

    all_data_dfs = []
    for data_id in fred_ids:
        logging.info(f'Pulling data for: "{data_id}" ')
        try:
            # Grab the data and make sure it is sorted by date
            data_df = fred.get_series(data_id).sort_index(ascending=True).reset_index()

            # Create a DataFrame from the data and add a column for the FRED ID
            data_df.columns = ['date', 'value']
            data_df['fred_id'] = data_id
            all_data_dfs.append(data_df)

            # Sleep a sliver of time to avoid hitting the API too hard
            time.sleep(SLEEP_TIME)

        except Exception as e:
            # Don't want to stop the entire process if one data series fails to pull
            # Log the error and continue with the next data series
            logging.error(f'Error pulling data for {data_id}. Continuing. Exception message: {e}. ')

    combo_data_df = pd.concat(all_data_dfs, ignore_index=True)
    combo_data_df['date'] = pd.to_datetime(combo_data_df['date']).dt.date

    if save_to_output:
        # Save the data to an Excel file
        combo_data_df.to_excel(FRED_DATA_DUMP_PATH, index=False)
        logging.info(f'Exported data to {FRED_DATA_DUMP_PATH}')
    
    return combo_data_df

def get_fred_data(fred_ids: list[str]=None, pull_new_data:bool=REFRESH_DATA) -> pd.DataFrame:
    """
    Get the FRED data for the given list of FRED IDs. If pull_new_data is True, it will pull new data from FRED.
    Otherwise, it will load the data from the Excel file. If no fred_ids, are provided, it will provide all data
    in the Excel file.
    """

    if pull_new_data:
        # Download the data from FRED
        combined_data_df = download_fred_data(fred_ids, save_to_output=True)
    else:
        # Load the data from the Excel file (Hopefully there is something there...)
        combined_data_df = pd.read_excel(FRED_DATA_DUMP_PATH)

    if fred_ids is not None:
        # Filter the data to only include the requested FRED IDs
        combined_data_df = combined_data_df[combined_data_df['fred_id'].isin(fred_ids)].copy()
    
    return combined_data_df


def create_fred_snapshot(pull_new_data=REFRESH_DATA) -> pd.DataFrame:

    fred_map_df = pd.read_excel(MASTER_FILE, sheet_name='master')
    
    # Grab the list of data points to include in this snapshot
    data_to_pull = fred_map_df[fred_map_df['dashboard_1'] == True]['fred_id'].to_list()

    if pull_new_data:
        # Download the data from FRED
        combined_data_df = download_fred_data(data_to_pull, save_to_output=True)
    else:
        # Load the data from the Excel file (Hopefully there is something there...)
        combined_data_df = pd.read_excel(FRED_DATA_DUMP_PATH, sheet_name='Sheet1')
    
     # Now loop through all of the data points and make a summary

    master_results = {}
    for data_series in combined_data_df['fred_id'].unique():
        this_data = combined_data_df[combined_data_df['fred_id'] == data_series]

        data_summary = {
            'latest': this_data.iloc[-1]['value'],
            'lag_1': this_data.iloc[-2]['value'],
            'lag_2': this_data.iloc[-3]['value'],
            'latest_date': this_data.iloc[-1]['date'],
            'lag_1_date': this_data.iloc[-2]['date'],
            'lag_2_date': this_data.iloc[-3]['date'],
        }
        master_results[data_series] = data_summary

    # Create a pretty DataFrame from the results
    results_df = pd.DataFrame(master_results).T
    results_df.columns = pd.MultiIndex.from_product([['Value', 'Date'],['latest', 'lag_1', 'lag_2']])

    # Add on the pretty name columns from master file
    results_df.index = results_df.index.map(fred_map_df.set_index('fred_id')['display_name'])  
    results_df.index.name = 'display_name'

    # If keeping below, need to change to use xlwings probably
    # results_df.to_excel(DASHBOARD_1_EXPORT_PATH, sheet_name='dashboard_1', index=True)
    # logging.info(f'Exported data to {DASHBOARD_1_EXPORT_PATH}')

    return results_df


if __name__ == "__main__":
    
    
    # create_fred_snapshot()
    create_fred_snapshot(pull_new_data=False)
    
    wb = xw.Book(DASHBOARD_1_EXPORT_PATH)
    print('Done!!!')
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\generate_report.py

```py
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import logging


import utils
import fred_fun as ff

# Paths
TEMPLATE_DIR = Path("templates")
OUTPUT_FOLDER = Path("output")
OUTPUT_HTML = Path("output") / "fred_dashboard_1.html"
# SNAPSHOT_FILE = Path("output") / "fred_dashboard_1.xlsx"
IMAGES_FOLDER = OUTPUT_FOLDER / "images"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generate_report.log'),
        logging.StreamHandler()
    ]
)


# Jinja env
jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

FORMAT_COLS = ['decimals', 'show_percent', 'use_commas', 'show_dollar']

master_fred_map_df = pd.read_excel(ff.MASTER_FILE, sheet_name='master')
utils.set_mpl_colors()

def get_macro_dashboard_data() -> list[dict]:

    # Grab the cleaned snapshot data    
    df = ff.create_fred_snapshot(pull_new_data=False)

    # Convert the datetime to a date
    df['Date'] = df['Date'].map(lambda x: x.date())

    fred_map_df = master_fred_map_df.set_index('display_name')
    rows = []
    for index, row in df.iterrows():
        meta_data = fred_map_df.loc[index]
        format_meta = meta_data[FORMAT_COLS].to_dict()
        rows.append({
            "display_name": index,
            "latest_value": utils.format_value(row["Value"]["latest"], **format_meta),
            "lag_1_value": utils.format_value(row["Value"]["lag_1"], **format_meta),
            "lag_2_value": utils.format_value(row["Value"]["lag_2"], **format_meta),
            "latest_date": row["Date"]["latest"],
            "lag_1_date": row["Date"]["lag_1"],
            "lag_2_date": row["Date"]["lag_2"],
            'url': meta_data['link'],
        })

    return rows




def generate_inflation_chart(inflation_df: pd.DataFrame) -> None:
    LOOKBACK_YEARS = 7
    df = inflation_df.iloc[-LOOKBACK_YEARS * 12:]

    # Map column names to display names
    name_mapper = master_fred_map_df.set_index('fred_id')['display_name']
    df.columns = df.columns.map(name_mapper)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)  # Larger size & higher resolution

    df.plot(ax=ax, linewidth=2)

    ax.set_title("12-Month Inflation", fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("")
    ax.set_ylabel("YoY % Change", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    fig.tight_layout()  # Removes extra whitespace
    chart_path = IMAGES_FOLDER / "inflation_chart.png"
    # fig.savefig(chart_path, bbox_inches="tight")
    fig.savefig(chart_path, bbox_inches="tight", transparent=True)

    plt.close(fig)



def generate_inflation_chart_plotly(inflation_df: pd.DataFrame) -> str:
    LOOKBACK_YEARS = 7
    df = inflation_df.iloc[-LOOKBACK_YEARS * 12:].copy()

    name_mapper = master_fred_map_df.set_index('fred_id')['display_name']
    df.columns = df.columns.map(name_mapper)
    df.index = pd.to_datetime(df.index)

    fig = go.Figure()

    for col in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col],
            mode='lines',
            name=col,
            hovertemplate='%{y:.2%}<extra>' + col + '</extra>',
        ))

    fig.update_layout(
        title="12-Month Inflation (YoY % Change)",
        title_font_size=20,
        height=500,
        width=900,
        xaxis_title="Date",
        yaxis_title="Year-over-Year Change",
        yaxis_tickformat=".0%",
        template="plotly_white",
        margin=dict(t=60, b=40, l=50, r=50),
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)"),
    )

    # Generate HTML div string
    chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    return chart_html


def generate_inflation_report() -> str:
    FRED_IDS = ['CPIAUCSL', 'CPILFESL', 'PCEPI', 'PCEPILFE']
    inflation_data = ff.get_fred_data(fred_ids=FRED_IDS)
    df = inflation_data.pivot(index='date', columns='fred_id', values='value')

    # % Changes
    df_1m = df.pct_change(1)
    df_3m = df.pct_change(3)
    df_6m = df.pct_change(6)
    df_12m = df.pct_change(12)

    fred_map = master_fred_map_df.set_index('fred_id')
    format_meta = {'show_percent': True, 'percent_convert': True}
    rows = []

    for fid in FRED_IDS:
        rows.append({
            'display_name': fred_map.loc[fid]['display_name'],
            'url': fred_map.loc[fid]['link'],
            'latest_date': df_12m[fid].last_valid_index().date(),
            'one_month': utils.format_value(df_1m[fid].dropna().iloc[-1], **format_meta),
            'three_month': utils.format_value(df_3m[fid].dropna().iloc[-1], **format_meta),
            'six_month': utils.format_value(df_6m[fid].dropna().iloc[-1], **format_meta),
            'twelve_month': utils.format_value(df_12m[fid].dropna().iloc[-1], **format_meta),
            'annualized_one_month': utils.format_value(df_1m[fid].dropna().iloc[-1] * 12, **format_meta),
            'annualized_three_month': utils.format_value(df_3m[fid].dropna().iloc[-1] * 4, **format_meta),
            'annualized_six_month': utils.format_value(df_6m[fid].dropna().iloc[-1] * 2, **format_meta),
        })

    # NEW: Generate interactive chart
    inflation_chart_html = generate_inflation_chart_plotly(df_12m)

    # Render HTML
    inflation_template = jinja_env.get_template("3_inflation.html")
    html = inflation_template.render(rows=rows, chart_html=inflation_chart_html)
    return html


def generate_inflation_report() -> str:


    FRED_IDS = ['CPIAUCSL', 'CPILFESL', 'PCEPI', 'PCEPILFE']
    # Pull the inflation data from FRED
    inflation_data = ff.get_fred_data(fred_ids=FRED_IDS)

    # Convert the dataframe to wide format to make it easier to work with
    df = inflation_data.pivot(index='date', columns='fred_id', values='value')


    # Compute % Changes
    df_1_month = df.pct_change(periods=1, fill_method=None)
    df_3_month = df.pct_change(periods=3, fill_method=None)
    df_6_month = df.pct_change(periods=6, fill_method=None)
    df_12_month = df.pct_change(periods=12, fill_method=None)

    # Generate the list of dictionarys for the data table
    fred_map = master_fred_map_df.set_index('fred_id')
    format_meta = {'show_percent': True, 'percent_convert': True}
    rows = []
    for data_id in FRED_IDS:
        # Get the date for the latest, non-null value
        row_dict = {}
        row_dict['display_name'] = fred_map.loc[data_id]['display_name']
        row_dict['url'] = fred_map.loc[data_id]['link']
        row_dict['latest_date'] = df_12_month[data_id].last_valid_index().date()
        row_dict['one_month'] = utils.format_value(df_1_month[data_id].dropna().iloc[-1], **format_meta)
        row_dict['three_month'] = utils.format_value(df_3_month[data_id].dropna().iloc[-1], **format_meta)
        row_dict['six_month'] = utils.format_value(df_6_month[data_id].dropna().iloc[-1], **format_meta)
        row_dict['twelve_month'] = utils.format_value(df_12_month[data_id].dropna().iloc[-1], **format_meta)
        row_dict['annualized_one_month'] = utils.format_value(df_1_month[data_id].dropna().iloc[-1] * 12, **format_meta)
        row_dict['annualized_three_month'] = utils.format_value(df_3_month[data_id].dropna().iloc[-1] * 4, **format_meta)
        row_dict['annualized_six_month'] = utils.format_value(df_6_month[data_id].dropna().iloc[-1] * 2, **format_meta)

        rows.append(row_dict)


    # generate_inflation_chart(df_12_month)
    chart_html = generate_inflation_chart_plotly(df_12_month)


    # Render the Jinja template with the inflation data
    inflation_template = jinja_env.get_template("3_inflation.html")
    # html = inflation_template.render(rows=rows)
    html = inflation_template.render(rows=rows, chart_html=chart_html)    

    return html 



def generate_report() -> None:

    logging.info("Beginning report generation...")

    full_report_template = jinja_env.get_template("0_full_report.html")
    dashboard_template = jinja_env.get_template("2_macro_dash.html")

    gdp_template = jinja_env.get_template("4_gdp.html")


    # Render individual sections to HTML snippets
    dashboard_html = dashboard_template.render(rows=get_macro_dashboard_data())
    gdp_html = gdp_template.render()
    inflation_html = generate_inflation_report()

    # Render full report with HTML snippets
    full_html = full_report_template.render(
        dashboard_section=dashboard_html,
        gdp_section=gdp_html,
        inflation_section=inflation_html,
    )    

    # Save output
    OUTPUT_HTML.parent.mkdir(exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(full_html)


    print(f"✅ Report generated at: {OUTPUT_HTML}")

if __name__ == "__main__":
    generate_report()
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\old.py

```py

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\README.md

```md
# Overview

Nothing here yet!!! Probably should be tho right.
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\utils.py

```py
import pandas as pd
from pathlib import Path
from matplotlib import cycler
import matplotlib as mpl
from matplotlib import pyplot as plt

def format_value(val, decimals=2, show_percent=False, use_commas=True, show_dollar=False,percent_convert=False) -> str:
    
    if pd.isnull(val):
        return "-"

    try:
        if percent_convert:
            val = val * 100

        # Build numeric format
        comma_flag = "," if use_commas else ""
        number_format = f"{{:{comma_flag}.{decimals}f}}"
        formatted = number_format.format(val)

        # Add dollar sign or percent symbol
        if show_dollar:
            formatted = f"${formatted}"
        if show_percent:
            formatted = f"{formatted}%"

        return formatted
    except Exception as e:
        return f"Error: {e}"




def set_mpl_colors() -> None:
    COLORS = [
        "#3f4c60",
        "#93c9f9",
        "#94045b",
        "#83889d",
        "#ffc000",
        "#386f98",
        "#9dabd3",
        "#b80571",
        "#45ad35",
        "#b38825",
        "#525e70",
        "#98bbdc",
        "#aa6597",
        "#6abd5d",
        "#716920",
    ]

    mpl.rcParams["axes.prop_cycle"] = cycler(color=COLORS)
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\reference\constants.py

```py

SINGLE_STOCK_TICKERS = ['AAPL','MSFT','AMZN','GOOGL','META','TSLA','JPM']
SECTOR_TICKERS = ['XLK', 'XLY', 'XLC', 'XLI', 'XLF', 'XLE', 'XLB', 'XLV', 'XLU', 'XLP','XLRE']
FI_TICKERS = ['AGG', 'BND', 'JBND']
MARKET_TICKERS = ['SPY', 'QQQ', 'IWM', 'VBR','ITOT']

MISC_TICKERS = []

ALL_TICKERS = SINGLE_STOCK_TICKERS + SECTOR_TICKERS + FI_TICKERS + MARKET_TICKERS + MISC_TICKERS
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\0_base.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Master Report</title>
    <style>
        body {
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            margin: 40px;
            background-color: #f9f9f9;
            color: #333;
        }
    
        h1 {
            font-size: 2em;
            border-bottom: 2px solid #222;
            padding-bottom: 0.3em;
        }
    
        table {
            border-collapse: collapse;
            width: 100%;
            background-color: #fff;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }
    
        th, td {
            padding: 10px 14px;
            border: 1px solid #e0e0e0;
            text-align: right;
            font-size: 0.95em;
        }
    
        td:first-child, th:first-child {
            text-align: left;
        }
    
        thead th {
            background-color: #f1f1f1;
            font-weight: bold;
            color: #333;
        }
    
        tbody tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    
        tbody tr:hover {
            background-color: #eaf3ff;
        }


    </style>


    
</head>

<body>
    <h1> Master Report</h1>
    {% block content %}{% endblock %}
</body>


</html>

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\0_full_report.html

```html
{% extends "0_base.html" %}

{% block content %}
    {{ dashboard_section | safe }}
    {{ gdp_section | safe }}
    {{ inflation_section | safe }}
{% endblock %}

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\2_macro_dash.html

```html
<h2> Macro Dashboard</h2>
   
<table>
    <thead>
        <tr>
            <th rowspan="2">Indicator</th>
            <th colspan="3">Value</th>
            <th colspan="3">Date</th>
        </tr>
        <tr>
            <th>Latest</th>
            <th>Lag 1</th>
            <th>Lag 2</th>
            <th>Latest</th>
            <th>Lag 1</th>
            <th>Lag 2</th>
        </tr>
    </thead>
    <tbody>
        {% for row in rows %}
        <tr>
            <td>
                <a href="{{ row.url }}" >{{ row.display_name }}</a>
            </td>
            <td>{{ row.latest_value }}</td>
            <td>{{ row.lag_1_value }}</td>
            <td>{{ row.lag_2_value }}</td>
            <td>{{ row.latest_date }}</td>
            <td>{{ row.lag_1_date }}</td>
            <td>{{ row.lag_2_date }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\3_inflation.html

```html
<h2>Inflation</h2>
<p>This section covers key inflation indicators.</p>


<div>
    <table>
        <thead>
            <tr>
                <th>Data</th>
                <th>Latest Date</th>
                <th>1-Month</th>
                <th>3-Month</th>
                <th>6-Month</th>
                <th>12-Month</th>
                <th>Annualized 1-Month</th>
                <th>Annualized 3-Month</th>
                <th>Annualized 6-Month</th>
            </tr>
        </thead>
            <tbody>
            {% for row in rows %}
            <tr>
                <td>
                    <a href="{{ row.url }}" >{{ row.display_name }}</a>
                </td>
                <td>{{ row.latest_date }}</td>
                <td>{{ row.one_month }}</td>
                <td>{{ row.three_month }}</td>
                <td>{{ row.six_month }}</td>
                <td>{{ row.twelve_month }}</td>
                <td>{{ row.annualized_one_month }}</td>
                <td>{{ row.annualized_three_month }}</td>
                <td>{{ row.annualized_six_month }}</td>
            </tr>
            {% endfor %}     

        </tbody>

    </table>

    <br>
    <!-- Add an image -->
    <!-- <img src="images/inflation_chart.png" alt="Inflation Chart" style="width: 100%; max-width: 1000px; height: auto; display: block; margin: 20px auto;"> -->
    
    <div>
        {{ chart_html | safe }}
    </div>

</div>
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\4_gdp.html

```html
<h2>GDP</h2>
<p>This section covers key GDP Data</p>

```

