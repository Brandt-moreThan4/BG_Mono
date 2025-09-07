# Project Master File

## Project Structure

```
├── .gitignore
├── .vscode
│   └── launch.json
├── README.md
├── bats
│   └── report_run_bg.bat
├── fred_snapshot.log
├── logs
│   └── generate_report.log
├── notebooks
│   ├── BG_Mono_master_project_document.md
│   ├── all_time_highs.ipynb
│   ├── db_connect.ipynb
│   └── master_markdown.ipynb
├── old.py
├── reference
│   ├── constants.py
│   └── fred.xlsx
├── sql
│   └── sm_ddl.sql
├── streamlit_app
│   ├── backtester.py
│   ├── data_engine.py
│   ├── home.py
│   ├── inputs.py
│   ├── metrics.py
│   ├── requirements.txt
│   ├── results.py
│   └── utils.py
├── utils.py
└── yahoo_finance.py
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\old.py

```py
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
import datetime
import logging

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


def make_date(input) -> datetime.date:
    """
    Convert whatever we have to a date object.
    """
    date_time = pd.to_datetime(input)
    return date_time.date()


def config_logging(log_file: str) -> None:
    """
    Configure logging to both file and console.
    """


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\yahoo_finance.py

```py
import pandas as pd
import yfinance as yf


class YFinanceData:
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self._raw_data_df = None
        self._price_data_df = None
        self._adjusted_price_data_df = None
        self._returns_df = None

        self.run()

    def fetch_data(self) -> None:
        self._raw_data_df = yf.download(self.tickers, group_by='ticker', auto_adjust=False, actions=False)
        self._raw_data_df.index = pd.to_datetime(self._raw_data_df.index)

    def clean_data(self) -> None:
        if self._raw_data_df is None:
            raise ValueError("No raw data to clean. Please fetch data first.")

        self._price_data_df = self._raw_data_df.loc[:, (slice(None), 'Close')]
        self._price_data_df.columns = self._price_data_df.columns.droplevel(1)

        self._adjusted_price_data_df = self._raw_data_df.loc[:, (slice(None), 'Adj Close')].copy()
        self._adjusted_price_data_df.columns = self._adjusted_price_data_df.columns.droplevel(1)
        self._adjusted_price_data_df.ffill(inplace=True)

        self._returns_df = self._adjusted_price_data_df.pct_change(fill_method=None)
        self._returns_df = self._returns_df[sorted(self._returns_df.columns)].copy()

    def run(self) -> None:
        self.fetch_data()
        self.clean_data()

    @property
    def raw_data(self) -> pd.DataFrame:
        if self._raw_data_df is None:
            raise ValueError("Raw data not available.")
        return self._raw_data_df.copy()

    @property
    def price_data(self) -> pd.DataFrame:
        if self._price_data_df is None:
            raise ValueError("Price data not available.")
        return self._price_data_df.copy()

    @property
    def adjusted_price_data(self) -> pd.DataFrame:
        if self._adjusted_price_data_df is None:
            raise ValueError("Adjusted price data not available.")
        return self._adjusted_price_data_df.copy()

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns_df is None:
            raise ValueError("Returns not available.")
        return self._returns_df.copy()

    def __repr__(self) -> str:
        return f"YFinanceData(tickers={self.tickers})"
    
    def __str__(self) -> str:
        return f"YFinanceData with {len(self.tickers)} tickers: {', '.join(self.tickers)}"


if __name__ == '__main__':
    TICKERS = ['AAPL', 'MSFT', 'GOOGL']
    yf_data = YFinanceData(TICKERS)
    returns_df = yf_data.returns
    print('Done!')

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\notebooks\BG_Mono_master_project_document.md

```md
# Project Master File

## Project Structure

```
├── .gitignore
├── .vscode
│   └── launch.json
├── README.md
├── bats
│   └── report_run_bg.bat
├── fred_snapshot.log
├── logs
│   └── generate_report.log
├── notebooks
│   ├── BG_Mono_master_project_document.md
│   ├── all_time_highs.ipynb
│   ├── db_connect.ipynb
│   └── master_markdown.ipynb
├── old.py
├── reference
│   ├── constants.py
│   └── fred.xlsx
├── sql
│   └── sm_ddl.sql
├── streamlit_app
│   ├── backtester.py
│   ├── data_engine.py
│   ├── home.py
│   ├── inputs.py
│   ├── metrics.py
│   ├── requirements.txt
│   ├── results.py
│   └── utils.py
├── utils.py
└── yahoo_finance.py
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\old.py

```py
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
import datetime
import logging

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


def make_date(input) -> datetime.date:
    """
    Convert whatever we have to a date object.
    """
    date_time = pd.to_datetime(input)
    return date_time.date()


def config_logging(log_file: str) -> None:
    """
    Configure logging to both file and console.
    """


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\yahoo_finance.py

```py
import pandas as pd
import yfinance as yf


class YFinanceData:
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self._raw_data_df = None
        self._price_data_df = None
        self._adjusted_price_data_df = None
        self._returns_df = None

        self.run()

    def fetch_data(self) -> None:
        self._raw_data_df = yf.download(self.tickers, group_by='ticker', auto_adjust=False, actions=False)
        self._raw_data_df.index = pd.to_datetime(self._raw_data_df.index)

    def clean_data(self) -> None:
        if self._raw_data_df is None:
            raise ValueError("No raw data to clean. Please fetch data first.")

        self._price_data_df = self._raw_data_df.loc[:, (slice(None), 'Close')]
        self._price_data_df.columns = self._price_data_df.columns.droplevel(1)

        self._adjusted_price_data_df = self._raw_data_df.loc[:, (slice(None), 'Adj Close')].copy()
        self._adjusted_price_data_df.columns = self._adjusted_price_data_df.columns.droplevel(1)
        self._adjusted_price_data_df.ffill(inplace=True)

        self._returns_df = self._adjusted_price_data_df.pct_change(fill_method=None)
        self._returns_df = self._returns_df[sorted(self._returns_df.columns)].copy()

    def run(self) -> None:
        self.fetch_data()
        self.clean_data()

    @property
    def raw_data(self) -> pd.DataFrame:
        if self._raw_data_df is None:
            raise ValueError("Raw data not available.")
        return self._raw_data_df.copy()

    @property
    def price_data(self) -> pd.DataFrame:
        if self._price_data_df is None:
            raise ValueError("Price data not available.")
        return self._price_data_df.copy()

    @property
    def adjusted_price_data(self) -> pd.DataFrame:
        if self._adjusted_price_data_df is None:
            raise ValueError("Adjusted price data not available.")
        return self._adjusted_price_data_df.copy()

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns_df is None:
            raise ValueError("Returns not available.")
        return self._returns_df.copy()

    def __repr__(self) -> str:
        return f"YFinanceData(tickers={self.tickers})"
    
    def __str__(self) -> str:
        return f"YFinanceData with {len(self.tickers)} tickers: {', '.join(self.tickers)}"


if __name__ == '__main__':
    TICKERS = ['AAPL', 'MSFT', 'GOOGL']
    yf_data = YFinanceData(TICKERS)
    returns_df = yf_data.returns
    print('Done!')

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\notebooks\BG_Mono_master_project_document.md

```md
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
├── generate_report.py
├── logs
│   └── generate_report.log
├── notebooks
│   ├── BG_Mono_master_project_document.md
│   ├── all_time_highs.ipynb
│   ├── db_connect.ipynb
│   └── master_markdown.ipynb
├── old.py
├── reference
│   ├── constants.py
│   └── fred.xlsx
├── sql
│   └── sm_ddl.sql
├── streamlit_app
│   ├── backtester.py
│   ├── data_engine.py
│   ├── home.py
│   ├── inputs.py
│   ├── metrics.py
│   ├── requirements.txt
│   ├── results.py
│   └── utils.py
├── utils.py
└── yahoo_finance.py
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
REFRESH_DATA = True  # Set to True to pull new data from FRED
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
import datetime
import shutil


import utils
import fred_fun as ff

# Paths
TEMPLATE_DIR = Path("templates")
OUTPUT_FOLDER = Path("output")
OUTPUT_HTML = OUTPUT_FOLDER / "Dashboard.html"
IMAGES_FOLDER = OUTPUT_FOLDER / "images"
CSS_TEMPLATE = TEMPLATE_DIR / 'css' / 'style.css'
CSS_OUTPUT_DIR = OUTPUT_FOLDER / 'css'


utils.config_logging('logs/generate_report.log')
utils.set_mpl_colors()

# Jinja env
jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

FORMAT_COLS = ['decimals', 'show_percent', 'use_commas', 'show_dollar']

master_fred_map_df = pd.read_excel(ff.MASTER_FILE, sheet_name='master')
    

def get_macro_dashboard_data() -> list[dict]:

    # Grab the cleaned snapshot data    
    df = ff.create_fred_snapshot(pull_new_data=ff.REFRESH_DATA)
    
    df['Date'] = df['Date'].map(lambda x: x if isinstance(x,datetime.date) else x.date())

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
            # Modified hovertemplate to include date
            hovertemplate='<b>Date: %{x|%b %d, %Y}</b><br>' +  # Added date with specific format
                          '<b>Value: %{y:.2%}</b><br>' +
                          '<extra>' + col + '</extra>',
        ))

    fig.update_layout(
        title="12-Month Inflation (YoY % Change)",
        title_font_size=20,
        height=500,
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Year-over-Year Change",
        yaxis_tickformat=".0%",
        template="plotly_white",
        margin=dict(t=60, b=40, l=50, r=50),
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)"),
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


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
        row_dict['latest_date'] = utils.make_date(df_12_month[data_id].last_valid_index())
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


    # Copy CSS file to output directory
    shutil.copy(CSS_TEMPLATE, CSS_OUTPUT_DIR / CSS_TEMPLATE.name)


    logging.info("Report generation complete.")



if __name__ == "__main__":
    generate_report()

    # Open the report
    import os
    os.startfile(OUTPUT_HTML)
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\old.py

```py
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
import datetime
import logging

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


def make_date(input) -> datetime.date:
    """
    Convert whatever we have to a date object.
    """
    date_time = pd.to_datetime(input)
    return date_time.date()


def config_logging(log_file: str) -> None:
    """
    Configure logging to both file and console.
    """


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\yahoo_finance.py

```py
import pandas as pd
import yfinance as yf


class YFinanceData:
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self._raw_data_df = None
        self._price_data_df = None
        self._adjusted_price_data_df = None
        self._returns_df = None

        self.run()

    def fetch_data(self) -> None:
        self._raw_data_df = yf.download(self.tickers, group_by='ticker', auto_adjust=False, actions=False)
        self._raw_data_df.index = pd.to_datetime(self._raw_data_df.index)

    def clean_data(self) -> None:
        if self._raw_data_df is None:
            raise ValueError("No raw data to clean. Please fetch data first.")

        self._price_data_df = self._raw_data_df.loc[:, (slice(None), 'Close')]
        self._price_data_df.columns = self._price_data_df.columns.droplevel(1)

        self._adjusted_price_data_df = self._raw_data_df.loc[:, (slice(None), 'Adj Close')].copy()
        self._adjusted_price_data_df.columns = self._adjusted_price_data_df.columns.droplevel(1)
        self._adjusted_price_data_df.ffill(inplace=True)

        self._returns_df = self._adjusted_price_data_df.pct_change(fill_method=None)
        self._returns_df = self._returns_df[sorted(self._returns_df.columns)].copy()

    def run(self) -> None:
        self.fetch_data()
        self.clean_data()

    @property
    def raw_data(self) -> pd.DataFrame:
        if self._raw_data_df is None:
            raise ValueError("Raw data not available.")
        return self._raw_data_df.copy()

    @property
    def price_data(self) -> pd.DataFrame:
        if self._price_data_df is None:
            raise ValueError("Price data not available.")
        return self._price_data_df.copy()

    @property
    def adjusted_price_data(self) -> pd.DataFrame:
        if self._adjusted_price_data_df is None:
            raise ValueError("Adjusted price data not available.")
        return self._adjusted_price_data_df.copy()

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns_df is None:
            raise ValueError("Returns not available.")
        return self._returns_df.copy()

    def __repr__(self) -> str:
        return f"YFinanceData(tickers={self.tickers})"
    
    def __str__(self) -> str:
        return f"YFinanceData with {len(self.tickers)} tickers: {', '.join(self.tickers)}"


if __name__ == '__main__':
    TICKERS = ['AAPL', 'MSFT', 'GOOGL']
    yf_data = YFinanceData(TICKERS)
    returns_df = yf_data.returns
    print('Done!')

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\notebooks\BG_Mono_master_project_document.md

```md
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
├── generate_report.py
├── logs
│   └── generate_report.log
├── notebooks
│   ├── BG_Mono_master_project_document.md
│   ├── all_time_highs.ipynb
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
│   ├── 4_gdp.html
│   └── css
│       └── style.css
├── utils.py
└── yahoo_finance.py
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
REFRESH_DATA = True  # Set to True to pull new data from FRED
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
import datetime
import shutil


import utils
import fred_fun as ff

# Paths
TEMPLATE_DIR = Path("templates")
OUTPUT_FOLDER = Path("output")
OUTPUT_HTML = OUTPUT_FOLDER / "Dashboard.html"
IMAGES_FOLDER = OUTPUT_FOLDER / "images"
CSS_TEMPLATE = TEMPLATE_DIR / 'css' / 'style.css'
CSS_OUTPUT_DIR = OUTPUT_FOLDER / 'css'


utils.config_logging('logs/generate_report.log')
utils.set_mpl_colors()

# Jinja env
jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

FORMAT_COLS = ['decimals', 'show_percent', 'use_commas', 'show_dollar']

master_fred_map_df = pd.read_excel(ff.MASTER_FILE, sheet_name='master')
    

def get_macro_dashboard_data() -> list[dict]:

    # Grab the cleaned snapshot data    
    df = ff.create_fred_snapshot(pull_new_data=ff.REFRESH_DATA)
    
    df['Date'] = df['Date'].map(lambda x: x if isinstance(x,datetime.date) else x.date())

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
            # Modified hovertemplate to include date
            hovertemplate='<b>Date: %{x|%b %d, %Y}</b><br>' +  # Added date with specific format
                          '<b>Value: %{y:.2%}</b><br>' +
                          '<extra>' + col + '</extra>',
        ))

    fig.update_layout(
        title="12-Month Inflation (YoY % Change)",
        title_font_size=20,
        height=500,
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Year-over-Year Change",
        yaxis_tickformat=".0%",
        template="plotly_white",
        margin=dict(t=60, b=40, l=50, r=50),
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)"),
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


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
        row_dict['latest_date'] = utils.make_date(df_12_month[data_id].last_valid_index())
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


    # Copy CSS file to output directory
    shutil.copy(CSS_TEMPLATE, CSS_OUTPUT_DIR / CSS_TEMPLATE.name)


    logging.info("Report generation complete.")



if __name__ == "__main__":
    generate_report()

    # Open the report
    import os
    os.startfile(OUTPUT_HTML)
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\old.py

```py
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
import datetime
import logging

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


def make_date(input) -> datetime.date:
    """
    Convert whatever we have to a date object.
    """
    date_time = pd.to_datetime(input)
    return date_time.date()


def config_logging(log_file: str) -> None:
    """
    Configure logging to both file and console.
    """


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\yahoo_finance.py

```py
import pandas as pd
import yfinance as yf


class YFinanceData:
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self._raw_data_df = None
        self._price_data_df = None
        self._adjusted_price_data_df = None
        self._returns_df = None

        self.run()

    def fetch_data(self) -> None:
        self._raw_data_df = yf.download(self.tickers, group_by='ticker', auto_adjust=False, actions=False)
        self._raw_data_df.index = pd.to_datetime(self._raw_data_df.index)

    def clean_data(self) -> None:
        if self._raw_data_df is None:
            raise ValueError("No raw data to clean. Please fetch data first.")

        self._price_data_df = self._raw_data_df.loc[:, (slice(None), 'Close')]
        self._price_data_df.columns = self._price_data_df.columns.droplevel(1)

        self._adjusted_price_data_df = self._raw_data_df.loc[:, (slice(None), 'Adj Close')].copy()
        self._adjusted_price_data_df.columns = self._adjusted_price_data_df.columns.droplevel(1)
        self._adjusted_price_data_df.ffill(inplace=True)

        self._returns_df = self._adjusted_price_data_df.pct_change(fill_method=None)
        self._returns_df = self._returns_df[sorted(self._returns_df.columns)].copy()

    def run(self) -> None:
        self.fetch_data()
        self.clean_data()

    @property
    def raw_data(self) -> pd.DataFrame:
        if self._raw_data_df is None:
            raise ValueError("Raw data not available.")
        return self._raw_data_df.copy()

    @property
    def price_data(self) -> pd.DataFrame:
        if self._price_data_df is None:
            raise ValueError("Price data not available.")
        return self._price_data_df.copy()

    @property
    def adjusted_price_data(self) -> pd.DataFrame:
        if self._adjusted_price_data_df is None:
            raise ValueError("Adjusted price data not available.")
        return self._adjusted_price_data_df.copy()

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns_df is None:
            raise ValueError("Returns not available.")
        return self._returns_df.copy()

    def __repr__(self) -> str:
        return f"YFinanceData(tickers={self.tickers})"
    
    def __str__(self) -> str:
        return f"YFinanceData with {len(self.tickers)} tickers: {', '.join(self.tickers)}"


if __name__ == '__main__':
    TICKERS = ['AAPL', 'MSFT', 'GOOGL']
    yf_data = YFinanceData(TICKERS)
    returns_df = yf_data.returns
    print('Done!')

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\notebooks\BG_Mono_master_project_document.md

```md
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
├── generate_report.py
├── logs
│   └── generate_report.log
├── notebooks
│   ├── BG_Mono_master_project_document.md
│   ├── all_time_highs.ipynb
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
│   ├── 4_gdp.html
│   └── css
│       └── style.css
├── utils.py
└── yahoo_finance.py
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
REFRESH_DATA = True  # Set to True to pull new data from FRED
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
import datetime
import shutil


import utils
import fred_fun as ff

# Paths
TEMPLATE_DIR = Path("templates")
OUTPUT_FOLDER = Path("output")
OUTPUT_HTML = OUTPUT_FOLDER / "Dashboard.html"
IMAGES_FOLDER = OUTPUT_FOLDER / "images"
CSS_TEMPLATE = TEMPLATE_DIR / 'css' / 'style.css'
CSS_OUTPUT_DIR = OUTPUT_FOLDER / 'css'


utils.config_logging('logs/generate_report.log')
utils.set_mpl_colors()

# Jinja env
jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

FORMAT_COLS = ['decimals', 'show_percent', 'use_commas', 'show_dollar']

master_fred_map_df = pd.read_excel(ff.MASTER_FILE, sheet_name='master')
    

def get_macro_dashboard_data() -> list[dict]:

    # Grab the cleaned snapshot data    
    df = ff.create_fred_snapshot(pull_new_data=ff.REFRESH_DATA)
    
    df['Date'] = df['Date'].map(lambda x: x if isinstance(x,datetime.date) else x.date())

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
            # Modified hovertemplate to include date
            hovertemplate='<b>Date: %{x|%b %d, %Y}</b><br>' +  # Added date with specific format
                          '<b>Value: %{y:.2%}</b><br>' +
                          '<extra>' + col + '</extra>',
        ))

    fig.update_layout(
        title="12-Month Inflation (YoY % Change)",
        title_font_size=20,
        height=500,
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Year-over-Year Change",
        yaxis_tickformat=".0%",
        template="plotly_white",
        margin=dict(t=60, b=40, l=50, r=50),
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)"),
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


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
        row_dict['latest_date'] = utils.make_date(df_12_month[data_id].last_valid_index())
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


    # Copy CSS file to output directory
    shutil.copy(CSS_TEMPLATE, CSS_OUTPUT_DIR / CSS_TEMPLATE.name)


    logging.info("Report generation complete.")



if __name__ == "__main__":
    generate_report()

    # Open the report
    import os
    os.startfile(OUTPUT_HTML)
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\old.py

```py
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
import datetime
import logging

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


def make_date(input) -> datetime.date:
    """
    Convert whatever we have to a date object.
    """
    date_time = pd.to_datetime(input)
    return date_time.date()


def config_logging(log_file: str) -> None:
    """
    Configure logging to both file and console.
    """


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\yahoo_finance.py

```py
import pandas as pd
import yfinance as yf


class YFinanceData:
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self._raw_data_df = None
        self._price_data_df = None
        self._adjusted_price_data_df = None
        self._returns_df = None

        self.run()

    def fetch_data(self) -> None:
        self._raw_data_df = yf.download(self.tickers, group_by='ticker', auto_adjust=False, actions=False)
        self._raw_data_df.index = pd.to_datetime(self._raw_data_df.index)

    def clean_data(self) -> None:
        if self._raw_data_df is None:
            raise ValueError("No raw data to clean. Please fetch data first.")

        self._price_data_df = self._raw_data_df.loc[:, (slice(None), 'Close')]
        self._price_data_df.columns = self._price_data_df.columns.droplevel(1)

        self._adjusted_price_data_df = self._raw_data_df.loc[:, (slice(None), 'Adj Close')].copy()
        self._adjusted_price_data_df.columns = self._adjusted_price_data_df.columns.droplevel(1)
        self._adjusted_price_data_df.ffill(inplace=True)

        self._returns_df = self._adjusted_price_data_df.pct_change(fill_method=None)
        self._returns_df = self._returns_df[sorted(self._returns_df.columns)].copy()

    def run(self) -> None:
        self.fetch_data()
        self.clean_data()

    @property
    def raw_data(self) -> pd.DataFrame:
        if self._raw_data_df is None:
            raise ValueError("Raw data not available.")
        return self._raw_data_df.copy()

    @property
    def price_data(self) -> pd.DataFrame:
        if self._price_data_df is None:
            raise ValueError("Price data not available.")
        return self._price_data_df.copy()

    @property
    def adjusted_price_data(self) -> pd.DataFrame:
        if self._adjusted_price_data_df is None:
            raise ValueError("Adjusted price data not available.")
        return self._adjusted_price_data_df.copy()

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns_df is None:
            raise ValueError("Returns not available.")
        return self._returns_df.copy()

    def __repr__(self) -> str:
        return f"YFinanceData(tickers={self.tickers})"
    
    def __str__(self) -> str:
        return f"YFinanceData with {len(self.tickers)} tickers: {', '.join(self.tickers)}"


if __name__ == '__main__':
    TICKERS = ['AAPL', 'MSFT', 'GOOGL']
    yf_data = YFinanceData(TICKERS)
    returns_df = yf_data.returns
    print('Done!')

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\notebooks\BG_Mono_master_project_document.md

```md
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
├── generate_report.py
├── logs
│   └── generate_report.log
├── notebooks
│   ├── BG_Mono_master_project_document.md
│   ├── all_time_highs.ipynb
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
│   ├── 4_gdp.html
│   └── css
│       └── style.css
├── utils.py
└── yahoo_finance.py
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
REFRESH_DATA = True  # Set to True to pull new data from FRED
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
import datetime
import shutil


import utils
import fred_fun as ff

# Paths
TEMPLATE_DIR = Path("templates")
OUTPUT_FOLDER = Path("output")
OUTPUT_HTML = OUTPUT_FOLDER / "Dashboard.html"
IMAGES_FOLDER = OUTPUT_FOLDER / "images"
CSS_TEMPLATE = TEMPLATE_DIR / 'css' / 'style.css'
CSS_OUTPUT_DIR = OUTPUT_FOLDER / 'css'


utils.config_logging('logs/generate_report.log')
utils.set_mpl_colors()

# Jinja env
jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

FORMAT_COLS = ['decimals', 'show_percent', 'use_commas', 'show_dollar']

master_fred_map_df = pd.read_excel(ff.MASTER_FILE, sheet_name='master')
    

def get_macro_dashboard_data() -> list[dict]:

    # Grab the cleaned snapshot data    
    df = ff.create_fred_snapshot(pull_new_data=ff.REFRESH_DATA)
    
    df['Date'] = df['Date'].map(lambda x: x if isinstance(x,datetime.date) else x.date())

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
            # Modified hovertemplate to include date
            hovertemplate='<b>Date: %{x|%b %d, %Y}</b><br>' +  # Added date with specific format
                          '<b>Value: %{y:.2%}</b><br>' +
                          '<extra>' + col + '</extra>',
        ))

    fig.update_layout(
        title="12-Month Inflation (YoY % Change)",
        title_font_size=20,
        height=500,
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Year-over-Year Change",
        yaxis_tickformat=".0%",
        template="plotly_white",
        margin=dict(t=60, b=40, l=50, r=50),
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)"),
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


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
        row_dict['latest_date'] = utils.make_date(df_12_month[data_id].last_valid_index())
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


    # Copy CSS file to output directory
    shutil.copy(CSS_TEMPLATE, CSS_OUTPUT_DIR / CSS_TEMPLATE.name)


    logging.info("Report generation complete.")



if __name__ == "__main__":
    generate_report()

    # Open the report
    import os
    os.startfile(OUTPUT_HTML)
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\old.py

```py
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
import datetime
import logging

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


def make_date(input) -> datetime.date:
    """
    Convert whatever we have to a date object.
    """
    date_time = pd.to_datetime(input)
    return date_time.date()


def config_logging(log_file: str) -> None:
    """
    Configure logging to both file and console.
    """


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\yahoo_finance.py

```py
import pandas as pd
import yfinance as yf


class YFinanceData:
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self._raw_data_df = None
        self._price_data_df = None
        self._adjusted_price_data_df = None
        self._returns_df = None

        self.run()

    def fetch_data(self) -> None:
        self._raw_data_df = yf.download(self.tickers, group_by='ticker', auto_adjust=False, actions=False)
        self._raw_data_df.index = pd.to_datetime(self._raw_data_df.index)

    def clean_data(self) -> None:
        if self._raw_data_df is None:
            raise ValueError("No raw data to clean. Please fetch data first.")

        self._price_data_df = self._raw_data_df.loc[:, (slice(None), 'Close')]
        self._price_data_df.columns = self._price_data_df.columns.droplevel(1)

        self._adjusted_price_data_df = self._raw_data_df.loc[:, (slice(None), 'Adj Close')].copy()
        self._adjusted_price_data_df.columns = self._adjusted_price_data_df.columns.droplevel(1)
        self._adjusted_price_data_df.ffill(inplace=True)

        self._returns_df = self._adjusted_price_data_df.pct_change(fill_method=None)
        self._returns_df = self._returns_df[sorted(self._returns_df.columns)].copy()

    def run(self) -> None:
        self.fetch_data()
        self.clean_data()

    @property
    def raw_data(self) -> pd.DataFrame:
        if self._raw_data_df is None:
            raise ValueError("Raw data not available.")
        return self._raw_data_df.copy()

    @property
    def price_data(self) -> pd.DataFrame:
        if self._price_data_df is None:
            raise ValueError("Price data not available.")
        return self._price_data_df.copy()

    @property
    def adjusted_price_data(self) -> pd.DataFrame:
        if self._adjusted_price_data_df is None:
            raise ValueError("Adjusted price data not available.")
        return self._adjusted_price_data_df.copy()

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns_df is None:
            raise ValueError("Returns not available.")
        return self._returns_df.copy()

    def __repr__(self) -> str:
        return f"YFinanceData(tickers={self.tickers})"
    
    def __str__(self) -> str:
        return f"YFinanceData with {len(self.tickers)} tickers: {', '.join(self.tickers)}"


if __name__ == '__main__':
    TICKERS = ['AAPL', 'MSFT', 'GOOGL']
    yf_data = YFinanceData(TICKERS)
    returns_df = yf_data.returns
    print('Done!')

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\notebooks\BG_Mono_master_project_document.md

```md
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
├── generate_report.py
├── logs
│   └── generate_report.log
├── notebooks
│   ├── BG_Mono_master_project_document.md
│   ├── all_time_highs.ipynb
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
│   ├── 4_gdp.html
│   └── css
│       └── style.css
├── utils.py
└── yahoo_finance.py
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
REFRESH_DATA = True  # Set to True to pull new data from FRED
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
import datetime
import shutil


import utils
import fred_fun as ff

# Paths
TEMPLATE_DIR = Path("templates")
OUTPUT_FOLDER = Path("output")
OUTPUT_HTML = OUTPUT_FOLDER / "Dashboard.html"
IMAGES_FOLDER = OUTPUT_FOLDER / "images"
CSS_TEMPLATE = TEMPLATE_DIR / 'css' / 'style.css'
CSS_OUTPUT_DIR = OUTPUT_FOLDER / 'css'


utils.config_logging('logs/generate_report.log')
utils.set_mpl_colors()

# Jinja env
jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

FORMAT_COLS = ['decimals', 'show_percent', 'use_commas', 'show_dollar']

master_fred_map_df = pd.read_excel(ff.MASTER_FILE, sheet_name='master')


def get_macro_dashboard_data() -> list[dict]:

    # Grab the cleaned snapshot data    
    df = ff.create_fred_snapshot(pull_new_data=ff.REFRESH_DATA)
    
    df['Date'] = df['Date'].map(lambda x: x if isinstance(x,datetime.date) else x.date())

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
            # Modified hovertemplate to include date
            hovertemplate='<b>Date: %{x|%b %d, %Y}</b><br>' +  # Added date with specific format
                          '<b>Value: %{y:.2%}</b><br>' +
                          '<extra>' + col + '</extra>',
        ))

    fig.update_layout(
        title="12-Month Inflation (YoY % Change)",
        title_font_size=20,
        height=500,
        autosize=True,
        xaxis_title="Date",
        yaxis_title="Year-over-Year Change",
        yaxis_tickformat=".0%",
        template="plotly_white",
        margin=dict(t=60, b=40, l=50, r=50),
        legend=dict(x=0, y=1, bgcolor="rgba(0,0,0,0)"),
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


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
        row_dict['latest_date'] = utils.make_date(df_12_month[data_id].last_valid_index())
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


    # Copy CSS file to output directory
    shutil.copy(CSS_TEMPLATE, CSS_OUTPUT_DIR / CSS_TEMPLATE.name)


    logging.info("Report generation complete.")



if __name__ == "__main__":
    generate_report()

    # Open the report
    import os
    os.startfile(OUTPUT_HTML)
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\old.py

```py
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
import datetime
import logging

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


def make_date(input) -> datetime.date:
    """
    Convert whatever we have to a date object.
    """
    date_time = pd.to_datetime(input)
    return date_time.date()


def config_logging(log_file: str) -> None:
    """
    Configure logging to both file and console.
    """


    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\yahoo_finance.py

```py
import pandas as pd
import yfinance as yf


class YFinanceData:
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self._raw_data_df = None
        self._price_data_df = None
        self._adjusted_price_data_df = None
        self._returns_df = None

        self.run()

    def fetch_data(self) -> None:
        self._raw_data_df = yf.download(self.tickers, group_by='ticker', auto_adjust=False, actions=False)
        self._raw_data_df.index = pd.to_datetime(self._raw_data_df.index)

    def clean_data(self) -> None:
        if self._raw_data_df is None:
            raise ValueError("No raw data to clean. Please fetch data first.")

        self._price_data_df = self._raw_data_df.loc[:, (slice(None), 'Close')]
        self._price_data_df.columns = self._price_data_df.columns.droplevel(1)

        self._adjusted_price_data_df = self._raw_data_df.loc[:, (slice(None), 'Adj Close')].copy()
        self._adjusted_price_data_df.columns = self._adjusted_price_data_df.columns.droplevel(1)
        self._adjusted_price_data_df.ffill(inplace=True)

        self._returns_df = self._adjusted_price_data_df.pct_change(fill_method=None)
        self._returns_df = self._returns_df[sorted(self._returns_df.columns)].copy()

    def run(self) -> None:
        self.fetch_data()
        self.clean_data()

    @property
    def raw_data(self) -> pd.DataFrame:
        if self._raw_data_df is None:
            raise ValueError("Raw data not available.")
        return self._raw_data_df.copy()

    @property
    def price_data(self) -> pd.DataFrame:
        if self._price_data_df is None:
            raise ValueError("Price data not available.")
        return self._price_data_df.copy()

    @property
    def adjusted_price_data(self) -> pd.DataFrame:
        if self._adjusted_price_data_df is None:
            raise ValueError("Adjusted price data not available.")
        return self._adjusted_price_data_df.copy()

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns_df is None:
            raise ValueError("Returns not available.")
        return self._returns_df.copy()

    def __repr__(self) -> str:
        return f"YFinanceData(tickers={self.tickers})"
    
    def __str__(self) -> str:
        return f"YFinanceData with {len(self.tickers)} tickers: {', '.join(self.tickers)}"


if __name__ == '__main__':
    TICKERS = ['AAPL', 'MSFT', 'GOOGL']
    yf_data = YFinanceData(TICKERS)
    returns_df = yf_data.returns
    print('Done!')

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\notebooks\BG_Mono_master_project_document.md

```md
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
    <link rel="stylesheet" href="css/style.css">
</head>

<body>
    <h1>Master Report</h1>
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

    <div style="text-align: center; margin: 20px 0;">
        <div style="max-width: 1200px; margin: 0 auto;">
            {{ chart_html | safe }}
        </div>
    </div>
    

</div>
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\4_gdp.html

```html
<h2>GDP</h2>
<p>This section covers key GDP Data</p>

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\css\style.css

```css
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

/* Add new styling for chart centering */
.chart-wrapper {
    text-align: center;
    margin: 20px 0;
}

.chart-inner {
    max-width: 1200px;
    margin: 0 auto;
}

```


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
    <link rel="stylesheet" href="css/style.css">
</head>

<body>
    <h1>Master Report</h1>
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

    <div style="text-align: center; margin: 20px 0;">
        <div style="max-width: 1200px; margin: 0 auto;">
            {{ chart_html | safe }}
        </div>
    </div>
    

</div>
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\4_gdp.html

```html
<h2>GDP</h2>
<p>This section covers key GDP Data</p>

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\css\style.css

```css
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

/* Add new styling for chart centering */
.chart-wrapper {
    text-align: center;
    margin: 20px 0;
}

.chart-inner {
    max-width: 1200px;
    margin: 0 auto;
}

```


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
    <link rel="stylesheet" href="css/style.css">
</head>

<body>
    <h1>Master Report</h1>
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

    <div style="text-align: center; margin: 20px 0;">
        <div style="max-width: 1200px; margin: 0 auto;">
            {{ chart_html | safe }}
        </div>
    </div>
    

</div>
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\4_gdp.html

```html
<h2>GDP</h2>
<p>This section covers key GDP Data</p>

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\css\style.css

```css
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

/* Add new styling for chart centering */
.chart-wrapper {
    text-align: center;
    margin: 20px 0;
}

.chart-inner {
    max-width: 1200px;
    margin: 0 auto;
}

```


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
    <link rel="stylesheet" href="css/style.css">
</head>

<body>
    <h1>Master Report</h1>
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

    <div style="text-align: center; margin: 20px 0;">
        <div style="max-width: 1200px; margin: 0 auto;">
            {{ chart_html | safe }}
        </div>
    </div>
    

</div>
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\4_gdp.html

```html
<h2>GDP</h2>
<p>This section covers key GDP Data</p>

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\templates\css\style.css

```css
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

/* Add new styling for chart centering */
.chart-wrapper {
    text-align: center;
    margin: 20px 0;
}

.chart-inner {
    max-width: 1200px;
    margin: 0 auto;
}

```


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

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\backtester.py

```py
import pandas as pd
import datetime
import numpy as np
import streamlit_app.data_engine as dd


# General, helper variables that are used later. 
all_dates = pd.date_range(start='1900-01-01',end='2099-12-31') # Big date range to cover all reasonable dates we may include in our backtest (This is filtered later)


class Backtester:

    pretty_name = 'BaseStrategy'
    short_name = 'BaseStrat'

    def __init__(
        self,
        data_blob: dd.DataEngine,
        tickers: list[str],
        weights: list[float],
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000,
        rebal_freq: str = 'QE',
        port_name: str = 'Port',
        params: dict = {}
    ) -> None:

        self.data_blob = data_blob
        self.rets_df = data_blob.rets_df
        self.input_tickers = tickers
        self.input_weights = weights
        self.port_name = port_name
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = start_date
        self.strat_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)] 
        self.rebalance_dates = pd.date_range(start=start_date,end=end_date,freq=rebal_freq) 
        # Make sure the end date is not included in the rebalance dates
        self.rebalance_dates = self.rebalance_dates[self.rebalance_dates != end_date]

        self.validate_data()


        self.portfolio = pd.Series(index=self.input_tickers,data=0.0,name=self.port_name)
        self.portfolio['Cash'] = initial_capital
        
        # Master dataframe to store the historical portfolio holdings
        self.portfolio_history_df = pd.DataFrame(index=self.strat_dates,columns=self.input_tickers)
        # self.portfolio_history_df.index.name

        # Just a catch all for any additional parameters that may be passed in for a substrategy
        self.params = params
    
    def validate_data(self) -> None:

        # Check that the input tickers are in the data blob
        for ticker in self.input_tickers:
            if ticker not in self.data_blob.tickers:
                raise ValueError(f'Ticker {ticker} not in data blob. Please check the input tickers.')

        # Check that the input weights sum to 1
        if np.abs(np.sum(self.input_weights) - 1) > 1e-8:
            raise ValueError('Input weights do not sum to 1. Please check the input weights.')

        

    def __repr__(self) -> str:
        return f'{self.short_name}: {self.start_date} - {self.end_date}'

    @property
    def port_value(self) -> float:
        return self.portfolio.sum()

    def rebalance_to_target_weights(self,target_weights:pd.Series) -> None:
        '''Rebalance the portfolio to the target weights provided. This will implictily involve selling off any
          securities that are overweight and buying any securities that are underweight.
    
        '''
    
        # Multiply the target weights by the current portfolio value to get the target value for each security
        target_values = target_weights * self.port_value

        # Update the new portfolio with the target values 
        # (This is implicitly carrying out trades...)
        self.portfolio = target_values
        self.portfolio_history_df.loc[self.current_date] = self.portfolio
    

    def increment_portfolio_by_returns(self) -> None:
        '''Increase the portfolio value by the returns for the current date. '''

        # If there is a return for the current date, then increment the portfolio by the returns
        if self.current_date in self.rets_df.index:
            self.portfolio = self.portfolio * (1 + self.rets_df.loc[self.current_date])
            
        # Regardless of if portfolio was incremented up or not, store the current portfolio value in the history for
        #  today's date. So we always have an estimated value for the portfolio at the end of each day.
        self.portfolio_history_df.loc[self.current_date] = self.portfolio

    
    def get_target_weights(self) -> pd.Series:
        '''Get the target weights for the portfolio based on the input weights. '''

        # Create a series with the input weights and the cash weight
        target_weights = pd.Series(index=self.input_tickers,data=self.input_weights)

        return target_weights

    def run_backtest(self,verbose=False) -> None:

        # Allocate the initial capital to the target weights
        target_weights = self.get_target_weights()
        self.rebalance_to_target_weights(target_weights)

        # Iterate through all the dates in the chosen time period
        for date in self.strat_dates[1:]:
            
            # Update the current date
            self.current_date = date

            # Increment the portfolio by the returns for the current date
            self.increment_portfolio_by_returns()

            # If the current date is a rebalance date, then rebalance the portfolio
            if date in self.rebalance_dates:
                if verbose:
                    print(f'Current Time {datetime.datetime.now()} Rebalancing: {date}')
                target_weights = self.get_target_weights()
                self.rebalance_to_target_weights(target_weights)

        # Calculate some useful data based on the portfolio history
        self.calculate_data()


    def calculate_data(self) -> None:
        '''Calculate some useful data based on the portfolio history which is nice to have when analyzing results.'''
        
        self.total_port_values = self.portfolio_history_df.sum(axis=1).astype(float).rename(self.port_name)
        self.weights_df = (self.portfolio_history_df.div(self.total_port_values,axis=0)).astype(float)
        self.wealth_index = self.total_port_values / self.total_port_values.iloc[0]

        self.cumulative_port_returns = self.wealth_index - 1

        # Calculate portfolio returns based on the total portfolio values
        self.portfolio_returns_all = self.total_port_values.pct_change().dropna()
        # self.portfolio_returns_all.name = self.short_name

        # But also calculate a portfolio return that removes days where the portfolio return is 0, because those
        # are with 99.999% certainty just holidays.
        basically_zero_mask = np.abs(self.portfolio_returns_all - 0) < 1e-8
        self.port_returns = self.portfolio_returns_all[~basically_zero_mask].copy()

        ''



if __name__ == '__main__':
    data = dd.DataEngine.load_saved_data() 
    bt = Backtester(data_blob=data,tickers=['AAPL','MSFT'],weights=[0.5,0.5],start_date='2010-01-01',end_date='2020-01-01')
    bt.run_backtest()
    print(bt.portfolio_history_df)

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\data_engine.py

```py
import os
import time
import pandas as pd
import yfinance as yf
import datetime as dt
import reference.constants as C
import streamlit as st

# DATA_FOLDER = 'data/'
DATA_FOLDER = 'temp_data/'
CACHE_EXPIRATION = 28800  # 8ish hours
MAX_FILES_SAVED = 100

class DataEngine:
    def __init__(self) -> None:
        self.adjusted_prices_df: pd.DataFrame = None
        self.rets_df: pd.DataFrame = None
        self.price_df: pd.DataFrame = None
        self.raw_data_df: pd.DataFrame = None

    def is_cache_expired(self, ticker: str) -> bool:
        file_path = f'{DATA_FOLDER}{ticker}.csv'
        if not os.path.exists(file_path):
            return True
        return (time.time() - os.path.getmtime(file_path)) > CACHE_EXPIRATION 

    def load_local_data(self, tickers: list[str]) -> pd.DataFrame:
        """Load data from local storage if available and not expired"""

        dfs = []
        for ticker in tickers:
            if self.is_cache_expired(ticker):
                # If any of the tickers are expired, return None (Meaning we will re-download everything)
                return None
            file_path = f'{DATA_FOLDER}{ticker}.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
                dfs.append(df)
        return pd.concat(dfs, axis=1) if dfs else None

    def check_storage_limit(self):
        """Checks if adding new files will exceed MAX_FILES_SAVED and clears folder if necessary"""
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)
            return
        
        existing_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
        if len(existing_files) >= MAX_FILES_SAVED:
            print("Storage limit exceeded, clearing folder...")
            for file in existing_files:
                os.remove(os.path.join(DATA_FOLDER, file))

    def save_data_locally(self, df: pd.DataFrame, tickers: list[str]) -> None:
        """Save downloaded data locally after checking storage limit"""
        self.check_storage_limit()
        os.makedirs(DATA_FOLDER, exist_ok=True)
        for ticker in tickers:
            mini_df = df[[ticker]].droplevel(0, axis=1).dropna()
            mini_df.to_csv(f'{DATA_FOLDER}{ticker}.csv')

    def download_new_data(self, tickers: list[str]) -> pd.DataFrame:
        tickers = list(dict.fromkeys(tickers))  # Remove duplicates
        # local_data = self.load_local_data(tickers)
        # if local_data is not None and not local_data.isnull().all().all():
        #     # print("Using cached data")
        #     self.raw_data_df = local_data
        # else:
        #     # print("Fetching new data from Yahoo Finance")
        self.raw_data_df = yf.download(tickers, group_by='ticker', auto_adjust=False, actions=False)
        self.save_data_locally(self.raw_data_df, tickers)

        self.clean_data()
        return self.rets_df

    def clean_data(self) -> pd.DataFrame:
        df = self.raw_data_df.copy()
        df.index = pd.to_datetime(df.index)
        
        self.price_df = df.loc[:, (slice(None), 'Close')]
        self.price_df.columns = self.price_df.columns.droplevel(1)

        adjusted_prices_df = df.loc[:, (slice(None), 'Adj Close')].copy()
        adjusted_prices_df.columns = adjusted_prices_df.columns.droplevel(1)
        adjusted_prices_df.ffill(inplace=True)
        
        self.rets_df = adjusted_prices_df.pct_change(fill_method=None)
        self.rets_df = self.rets_df[sorted(self.rets_df.columns)].copy()
        self.adjusted_prices_df = adjusted_prices_df
        
        return self.rets_df

    def save_data(self, folder_path=DATA_FOLDER) -> None:
        os.makedirs(folder_path, exist_ok=True)
        self.rets_df.to_csv(f'{folder_path}rets_df.csv')
        self.adjusted_prices_df.to_csv(f'{folder_path}adjusted_prices_df.csv')
        self.price_df.to_csv(f'{folder_path}price_df.csv')

    @staticmethod
    def load_saved_data(folder: str = DATA_FOLDER) -> "DataEngine":
        dblob = DataEngine()
        dblob.rets_df = pd.read_csv(f'{folder}rets_df.csv', index_col=0, parse_dates=True)
        dblob.adjusted_prices_df = pd.read_csv(f'{folder}adjusted_prices_df.csv', index_col=0, parse_dates=True)
        dblob.price_df = pd.read_csv(f'{folder}price_df.csv', index_col=0, parse_dates=True)
        dblob.price_df.index = pd.to_datetime(dblob.price_df.index)
        return dblob
    
    @property
    def tickers(self) -> list[str]:
        return self.rets_df.columns.tolist()

if __name__ == '__main__':
    downloader = DataEngine()
    downloader.download_new_data(C.MARKET_TICKERS)
    downloader.load_local_data(C.MARKET_TICKERS)
    print('Done!')

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\home.py

```py
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as dt

import inputs as inputs
import data_engine as dd
import backtester as bt
import results as rs

st.title("Portfolio Backtester")
html_title = """
<h4 style="font-family: 'Arial', sans-serif; font-size: 1.2rem; font-weight: bold; font-style: italic; color: #555; margin-top: -10px; margin-bottom: 20px; text-shadow: 1px 1px #ddd;">A Brandt Green Production</h4>
"""
st.markdown(html_title, unsafe_allow_html=True)

st.sidebar.markdown("## Table of Contents")
st.sidebar.markdown("""
- [Inputs](#inputs)
- [Results](#results)
  - [Cumulative Returns](#cumulative-returns)
  - [Volatility](#volatility)    
  - [Performance Metrics](#performance-metrics)                
  - [Correlation Matrix](#correlation-matrix)                                        
  - [Portfolio Weights Over Time](#portfolio-weights-over-time)
  - [Individual Prices](#individual-prices)
- [Raw Data Reference](#raw-data-reference)
  - [Rebalance Dates](#rebalance-dates)                    
  - [Raw Returns](#raw-returns)
  - [Raw Port Weights](#raw-portfolio-weights)
""", unsafe_allow_html=True)




# ----------------------------
# Collect User Inputs
# ----------------------------

cleaned_inputs = inputs.get_user_inputs()

# run_backtest = st.button("Run Backtest")


# ----------------------------
# Fetch Market Data & Validate against Inputs
# ----------------------------

needed_tickers = list(dict.fromkeys(cleaned_inputs.tickers + [cleaned_inputs.bench_ticker]))


# Need to uncomment out below in a bit
# if not run_backtest:
#     st.stop()


with st.spinner("Fetching data..."):
    data = dd.DataEngine()
    
    # Try loading cached data first
    data.raw_data_df = data.load_local_data(needed_tickers)
    
    # May need to review below to fetch data for any new tickers
    if data.raw_data_df is None or cleaned_inputs.fetch_new_data:
        # st.warning("Fetching new data from Yahoo Finance. This may take a second...")
        with st.spinner("Fetching new data from Yahoo Finance. This may take a second..."):
            data.download_new_data(needed_tickers)
    
    # Make sure it's been cleaned
    data.clean_data()


# Validate we have the data to run a backtest
# Ensure selected tickers exist in dataset (Should be moved somewhere else???)
missing_tickers = [t for t in cleaned_inputs.tickers if t not in data.tickers]
if missing_tickers:
    error_msg = f"""Missing data for some tickers. Sorry... If you want to fetch new data, toggle the buttom
\n Missing tickers: {missing_tickers}"""
    st.error(error_msg)
    st.stop()


# Filter returns dataframe for only the selected tickers
data.rets_df = data.rets_df[needed_tickers].copy()



# Check that we have returns for all tickers for the entire backtest period
missing_returns = data.rets_df.loc[cleaned_inputs.start_date:cleaned_inputs.end_date].isnull().sum()
if missing_returns.any():
    error_msg = f"""Missing returns for some tickers during the backtest period. Sorry... 
\n Problem tickers: {missing_returns[missing_returns > 0].index.tolist()}"""
    st.error(error_msg)
    st.stop()



# ----------------------------
# Run Backtest
# ----------------------------

with st.spinner("Running backtest..."):
    backtester = bt.Backtester(
        data_blob=data,
        tickers=cleaned_inputs.tickers,
        weights=cleaned_inputs.weights,
        start_date=str(cleaned_inputs.start_date),
        end_date=str(cleaned_inputs.end_date),
        rebal_freq=cleaned_inputs.rebalance_freq,
        port_name=cleaned_inputs.port_name,
    )
    backtester.run_backtest()

# ----------------------------
# Display Results
# ----------------------------

# Add a few line breaks and a separator to distinguish the results section
st.markdown("---")
st.markdown("## Results")

rs.display_results(backtester, data, cleaned_inputs)

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\inputs.py

```py
from dataclasses import dataclass
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as dt
import reference.constants as C
from 
from utils import DynamicDates

# Add a dataclass to hold the user inputs


@dataclass
class CleanInputs:

    tickers: list
    weights: str
    start_date: dt.datetime
    end_date: dt.datetime
    port_name: str
    rebalance_freq: str
    bench_ticker: str
    fetch_new_data: bool = False



def get_user_inputs():
    """Collects user inputs and returns them as variables."""

    st.markdown("## Inputs")

    # ------------------
    # Ticker Input
    # ------------------

    st.markdown("#### Tickers")
    default_ticks = ' '.join(C.SECTOR_TICKERS)
    tickers_input = st.text_area(
        "Enter tickers separated by spaces (e.g., AAPL MSFT AMZN GOOGL META TSLA JPM):",
        default_ticks
    )

    tickers = [t.strip().upper() for t in tickers_input.split(" ") if t.strip()]

    TICKER_LIMIT = 50
    if len(tickers) > TICKER_LIMIT:
        st.error(f'Sorry, for now the maximum tickers allowed is {TICKER_LIMIT}. Because I am worried about abusing the API. ')
        st.stop()

    # Raise error if there are duplicates
    if len(tickers) != len(set(tickers)):
        dups = set([ticker for ticker in tickers if tickers.count(ticker) > 1])
        st.error(f"Duplicate tickers found. Please remove: {dups}")

    # ------------------
    # Weights Input
    # ------------------
    st.markdown("#### Portfolio Weights")

    equal_weights = [1 / len(tickers) * 100]  * len(tickers)
    equal_weights = [f'{round(w, 2)}' for w in equal_weights]
    equal_weights_str = " ".join(equal_weights)
    weights_msg = "Enter the target weights for each ticker. Defaults to equal-weight. Space-separated. Percentagses. Should sum to 1, e.g, 35 25 40:"
    weights_input = st.text_area(
        weights_msg,
        equal_weights_str
    )

    weights_input = weights_input.split(" ")
    # Make sure the number of weights matches the number of tickers
    if len(weights_input) != len(tickers):
        st.error(f"Number of weights does not match number of tickers. Please provide a weight for each ticker.")
        st.stop()

    # Convert to floats
    weights_input = [float(w)/100 for w in weights_input]

    # Validat that weights are closish to 1

    DIFF_THRESHOLD = .05
    if abs(1 - sum(weights_input)) > DIFF_THRESHOLD:

        st.error(f"Your weights do not sum to 1. Please ensure they sum to 1. Current sum: {sum(weights_input)}")
        st.stop()
    
    # Rescale the weights anyway (to handle when they are super close)
    weights_input = [w / sum(weights_input) for w in weights_input]

    # ------------------
    # Date Selection
    # ------------------

    st.markdown("#### Date Range")

    # Date range selection dropdown
    date_option = st.selectbox(
        "Select a lookback range (this just assists in picking start date):",
        ["Custom", "1D", "YTD", "1 Year", "3 Years", "5 Years", "10 Years", "15 Years"],
        index=2,  # Default to "YTD"
    )

    # Automatically update start date based on selection
    date_dict = {
        "1D": DynamicDates.day_before_yesterday(),
        "YTD": DynamicDates.prior_year_end(),
        "1 Year": DynamicDates.one_year_ago(),
        "3 Years": DynamicDates.three_years_ago(),
        "5 Years": DynamicDates.five_years_ago(),
        "10 Years": DynamicDates.ten_years_ago(),
        "15 Years": DynamicDates.fifteen_years_ago(),
    }


    start_date_default = date_dict.get(date_option,None)
    if start_date_default is None:
        # If the default is none, 
        start_date_default = DynamicDates.day_before_yesterday()

    # Start date (users can override)
    start_date = st.date_input("Start Date (assumes you invest at close of this date):", start_date_default)

    # End Date Selection
    # if date_option != 'Custom':
    end_date = st.date_input("End Date (assumes you liquidate at close of this date):", DynamicDates.yesterday())
    # else:
    #     end_date = st.date_input("End Date (Assumes you liquidate at close of this date):")
    # ------------------
    # Rebalance Frequency Selection
    # ------------------

    st.markdown("#### Rebalancing Options")
    rebalance_freq = st.selectbox(
        "Select rebalance frequency:",
        [
            "YE - Year end",
            "YS - Year start",
            "QS - Quarter start",
            "QE - Quarter end",
            "MS - Month start",
            "ME - Month end",
            "W - Weekly",
            "D - Calendar day",
        ],
        index=3  # Default to "QE - Quarter end"
    )

    rebalance_freq = rebalance_freq.split(" - ")[0]  # Extract alias


    # ------------------
    # Benchmark Selection
    # ------------------
    st.markdown("#### Benchmark")
    benchmark = st.selectbox(
        "Select a benchmark (only used to calculate beta):",
        ["SPY", "IWM","QQQ","BND"],
        index=0
    )



    # ------------------
    # Portfolio Name
    # ----------------
    port_name = st.text_input("Enter a name for your portfolio:", "Port")
    # st.markdown('---')

    # Add a toggle to fetch new date or use old
    # st.markdown("#### Force Fetch New Data")
    fetch_new_data = False
    # st.write("If you need to fetch new data (not using data that is cached), toggle the switch below. You should probably never need to do this.")
    # fetch_new_data = st.toggle("Query Updated Data", value=False)

    
    
    clean_inputs = CleanInputs(
        tickers=tickers,
        weights=weights_input,
        start_date=start_date,
        end_date=end_date,
        port_name=port_name,
        rebalance_freq=rebalance_freq,
        bench_ticker=benchmark,
        fetch_new_data=fetch_new_data
    )
    return clean_inputs
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\metrics.py

```py
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

def calculate_beta(rets:pd.Series,bench_rets:pd.Series) -> float:
    '''Calculate the beta of the portfolio returns against the benchmark returns.'''
    
    # Join series together to have matching dates
    both_rets = pd.concat([rets,bench_rets],axis=1).dropna()

    cov_matrix = both_rets.cov()
    beta = cov_matrix.iloc[0,1] / cov_matrix.iloc[1,1]

    return beta

def calculate_alpha(returns:pd.Series,bench_rets:pd.Series) -> float:
    '''Calculate annualzied alpha.'''

    # Create a cleaneddataframe so we can feed it into statsmodels
    data = pd.concat([returns,bench_rets],axis=1).dropna()
    data.columns = ['port','bench']
    model = smf.ols('port ~ bench',data=data).fit()
    alpha = model.params['Intercept'] * 252

    return alpha


def upside_capture(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    '''Calculate the upside capture ratio using average returns.'''
    
    # Join series together to align dates
    both_rets = pd.concat([port_rets, bench_rets], axis=1).dropna()
    up_market_rets = both_rets[both_rets.iloc[:, 1] > 0]  # Use only periods where the benchmark is positive

    # Calculate average returns
    port_avg_ret = up_market_rets.iloc[:, 0].mean()
    bench_avg_ret = up_market_rets.iloc[:, 1].mean()

    # Avoid division by zero
    if bench_avg_ret == 0:
        return np.nan

    # Calculate upside capture ratio
    up_capture = port_avg_ret / bench_avg_ret
    return up_capture


def downside_capture(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    '''Calculate the downside capture ratio using average returns.'''
    
    # Join series together to align dates
    both_rets = pd.concat([port_rets, bench_rets], axis=1).dropna()
    down_market_rets = both_rets[both_rets.iloc[:, 1] < 0]  # Use only periods where the benchmark is negative

    # Calculate average returns
    port_avg_ret = down_market_rets.iloc[:, 0].mean()
    bench_avg_ret = down_market_rets.iloc[:, 1].mean()

    # Avoid division by zero
    if bench_avg_ret == 0:
        return np.nan

    # Calculate downside capture ratio
    down_capture = port_avg_ret / bench_avg_ret
    return down_capture


def get_downside_deviation(returns, target=0):
    downside_diff = np.maximum(0, target - returns)
    squared_diff = np.square(downside_diff)
    mean_squared_diff = np.nanmean(squared_diff)
    dd = np.sqrt(mean_squared_diff) * np.sqrt(252)
    return dd


def get_max_drawdown(returns:pd.Series) -> float:

    wealth_index = (1 + returns).cumprod().array

    # Insert a wealth index of 1 at the beginning to make the calculation work
    wealth_index = np.insert(wealth_index,0,1)
    # Get the cumulative max
    cum_max = np.maximum.accumulate(wealth_index)
    max_dd = ((wealth_index / cum_max) - 1).min()

    return max_dd


def calculate_metrics(returns:pd.Series,bench_rets:pd.Series) -> dict:
    '''Calculate the key metrics for a given series of returns. Assumes returns are daily.'''

    total_ret = (1+returns).prod() - 1
    cagr = (total_ret + 1) ** (252 / returns.shape[0]) - 1
    vol = returns.std() * np.sqrt(252)
    max_dd = get_max_drawdown(returns)

    # Sharpe Ratio
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    alpha = calculate_alpha(returns,bench_rets)


    beta = calculate_beta(returns,bench_rets)
    downside_deviation = get_downside_deviation(returns)

    up_capture = upside_capture(returns,bench_rets)
    down_capture = downside_capture(returns,bench_rets)

    metrics = {
        'Total Return': total_ret,
        'CAGR': cagr,
        'Volatility': vol,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Beta': beta,
        'Alpha': alpha,
        'Downside Deviation': downside_deviation,
        'Up Capture': up_capture,
        'Down Capture': down_capture

    }

    return pd.Series(metrics)

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\requirements.txt

```txt
pandas
numpy
datetime
yfinance
streamlit
plotly
statsmodels
matplotlib

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\results.py

```py
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as dt

import streamlit_app.data_engine as dd
import backtester as bt
import streamlit_app.inputs as inputs
import metrics
import utils


# Utility Functions
def format_as_percent(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Format specified columns as percentages."""
    for column in columns:
        df[column] = df[column].map('{:.2%}'.format)
    return df


def plot_line_chart(df: pd.DataFrame, title: str, yaxis_title: str) -> None:
    fig = px.line(df, title=title)
    fig.update_yaxes(tickformat=".2%", title_text=yaxis_title)
    fig.update_xaxes(title_text="Date")
    st.plotly_chart(fig)


def plot_bar_chart(df: pd.Series, title: str, yaxis_title: str) -> None:
    fig = px.bar(df, title=title)
    fig.update_yaxes(tickformat=".2%", title_text=yaxis_title)
    fig.update_xaxes(title_text="")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)


def display_results(backtest:bt.Backtester,data:dd.DataEngine, cleaned_inputs:inputs.CleanInputs) -> None:

    start_dt = pd.to_datetime(cleaned_inputs.start_date)

    # Security returns should be after the start date (non inclusive) and before the end date (inclusive)
    rets_filered_df = data.rets_df[data.rets_df.index > start_dt].loc[:cleaned_inputs.end_date]
    security_rets_df = rets_filered_df[cleaned_inputs.tickers]
    bench_rets = rets_filered_df[cleaned_inputs.bench_ticker]
    all_rets_df = pd.concat([backtest.port_returns, security_rets_df], axis=1)
    security_prices_df = data.price_df[data.price_df.index > start_dt].loc[:cleaned_inputs.end_date]
    security_prices_df = security_prices_df[cleaned_inputs.tickers]

    bench_rets = rets_filered_df[cleaned_inputs.bench_ticker]

    st.markdown("### Cumulative Returns")

    cum_rets_df = (1 + all_rets_df).cumprod() - 1

    plot_line_chart(cum_rets_df, "Cumulative Returns", "Cumulative Returns")
    
    # Bar plot of total return
    total_rets = cum_rets_df.iloc[-1].sort_values(ascending=False)
    plot_bar_chart(total_rets, "Total Returns", "Total Return")

    # Add on yearly returns, if we have enough data
    annual_rets = all_rets_df.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    if len(annual_rets) > 1:
        st.markdown("#### Annual Returns")        
        annual_rets.index = annual_rets.index.year
        annual_rets = annual_rets.T
        # Format these returns as a heatmap each year
        annual_rets = annual_rets.style.format("{:.2%}").background_gradient(cmap='RdYlGn', axis=1)


        st.write(annual_rets)

    # Volatility
    # Display the vol in a bar chart in the same order as the total rets
    st.markdown("### Volatility")    
    total_vol = all_rets_df.std() * 252 ** 0.5

    total_vol = total_vol[total_rets.index]
    plot_bar_chart(total_vol, "Total Period Annualized Volatility", "Volatility")

    # If you have enough data, plot the rolling vol
    ROLLING_WINDOW = 252
    if len(all_rets_df) > ROLLING_WINDOW:
    
        rolling_vols = all_rets_df.rolling(window=252).std() * 252 ** 0.5
        rolling_vols = rolling_vols.dropna()
        plot_line_chart(rolling_vols, "Rolling 1-Year Volatility", "Volatility")

    #---------------------------
    # Metrics
    #---------------------------

    st.markdown("### Performance Metrics")    
    metrics_df = all_rets_df.apply(metrics.calculate_metrics, args=(bench_rets,),axis=0)
    # We want to apply lots of fun formatting to the metrics
    metrics_pretty_df = metrics_df.T.copy()
    COLS_TO_PRETTIFY = ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown', 'Alpha', 'Downside Deviation']
    metrics_pretty_df = format_as_percent(metrics_pretty_df, COLS_TO_PRETTIFY)
    # Format the following columns to onlu 2 decimal places
    DECIMAL_COLS = ['Beta', 'Sharpe', 'Up Capture', 'Down Capture']
    metrics_pretty_df[DECIMAL_COLS] = metrics_pretty_df[DECIMAL_COLS].map('{:.2f}'.format)

    st.write(metrics_pretty_df)

    # ----------------------------
    # Correlation Matrix
    # ----------------------------
    st.markdown("### Correlation Matrix")
    corr = all_rets_df.corr()
    corr_pretty_df = corr.copy()
    # # corr_pretty_df = corr_pretty_df.applymap('{:.2f}'.format)
    corr_pretty_df = corr.style.format("{:.2f}").background_gradient(cmap='coolwarm', vmin=-1, vmax=1)
    st.write(corr_pretty_df)


    # Portfolio Weights Over Time
    st.markdown("### Portfolio Weights Over Time")
    plot_line_chart(backtest.weights_df, "Portfolio Weights Over Time", "Weight")


    # # Returns
    # st.markdown("### Individual Returns")
    # # Add a tab for each of the securities and show a plot of it's indivdiual cumualtive return
    # ret_tabs = st.tabs(cum_rets_df.columns.to_list())
    # for ticker, tab in zip(cum_rets_df.columns, ret_tabs):
    #     with tab:
    #         total_ret = cum_rets_df[ticker].iloc[-1]
    #         st.write(f"Total Return: {total_ret:.2%}")
    #         plot_line_chart(cum_rets_df[ticker], f"{ticker} Cumulative Return", "Cumulative Return")

    #         # Add on another bar chart that shows the return on different time periods.


    st.markdown("### Individual Prices")
    st.write('_These prices are not adjusted for splits or dividends_')
    price_tabs = st.tabs(security_prices_df.columns.to_list())
    for ticker, tab in zip(security_prices_df.columns, price_tabs):
        with tab:
            fig = px.line(security_prices_df[ticker], title=f"{ticker} Prices")
            # Format in dollars
            fig.update_yaxes(title_text="Price", tickprefix="$")
            fig.update_xaxes(title_text="Date")
            st.plotly_chart(fig)


    

    # # ----------------------------
    # # Display Raw Data
    # # ----------------------------

    st.markdown("## Raw Data Reference")

    st.markdown("### Rebalance Dates")
    dates = backtest.rebalance_dates
    dates.name = 'Rebalance Dates'
    dates = pd.Series(dates.date, name='Rebalance Dates')
    st.write(dates)



    st.markdown("### Raw Returns")
    rets_df = utils.convert_dt_index(all_rets_df)
    # Format the returns as percentages and color code them. Positive returns are green, negative are red.
    styled_df = rets_df.style.format("{:.2%}").map(utils.color_returns)
    st.write(styled_df)

    # st.markdown("### Portfolio History")
    # st.write(utils.convert_dt_index(backtest.portfolio_history_df))

    st.markdown("### Raw Portfolio Weights")
    weights_df = utils.convert_dt_index(backtest.weights_df)
    weights_df = weights_df.map('{:.2%}'.format)
    st.write(weights_df)


if __name__ == '__main__':

    data = dd.DataEngine.load_saved_data() 
    back = bt.Backtester(data_blob=data,tickers=['AAPL','MSFT'],weights=[0.5,0.5],start_date='2010-01-01',end_date='2020-01-01')

    back.run_backtest()


    cleaned_inputs = inputs.CleanInputs(tickers=['AAPL','MSFT'],weights='0.5,0.5',start_date=dt.datetime(2010,1,1),end_date=dt.datetime(2020,1,1),port_name='Portfolio',rebalance_freq='QE',fetch_new_data=False,bench_ticker='SPY')

    display_results(back,data,cleaned_inputs)
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\utils.py

```py
import pandas as pd
import datetime as dt

def convert_dt_index(df:pd.DataFrame) -> pd.DataFrame:
    '''Convert the index of a DataFrame from datetime to date'''
    df.index = pd.to_datetime(df.index).date
    return df

def color_returns(val):
    color = "green" if val > 0 else "red"
    return f"color: {color}"


class DynamicDates:
    @classmethod
    def today(cls):
        return dt.datetime.today()

    @classmethod
    def yesterday(cls):
        return cls.today() - dt.timedelta(days=1)

    @classmethod
    def day_before_yesterday(cls):
        return cls.today() - dt.timedelta(days=2)

    @classmethod
    def prior_year_end(cls):
        return dt.datetime(cls.today().year - 1, 12, 31)

    @classmethod
    def one_year_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 1)

    @classmethod
    def three_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 3)

    @classmethod
    def five_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 5)

    @classmethod
    def ten_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 10)

    @classmethod
    def fifteen_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 15)


if __name__ == '__main__':
    print("Today:", DynamicDates.today())
    print("Yesterday:", DynamicDates.yesterday())
    print("One Year Ago:", DynamicDates.one_year_ago())

```


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

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\backtester.py

```py
import pandas as pd
import datetime
import numpy as np
import streamlit_app.data_engine as dd


# General, helper variables that are used later. 
all_dates = pd.date_range(start='1900-01-01',end='2099-12-31') # Big date range to cover all reasonable dates we may include in our backtest (This is filtered later)


class Backtester:

    pretty_name = 'BaseStrategy'
    short_name = 'BaseStrat'

    def __init__(
        self,
        data_blob: dd.DataEngine,
        tickers: list[str],
        weights: list[float],
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000,
        rebal_freq: str = 'QE',
        port_name: str = 'Port',
        params: dict = {}
    ) -> None:

        self.data_blob = data_blob
        self.rets_df = data_blob.rets_df
        self.input_tickers = tickers
        self.input_weights = weights
        self.port_name = port_name
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = start_date
        self.strat_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)] 
        self.rebalance_dates = pd.date_range(start=start_date,end=end_date,freq=rebal_freq) 
        # Make sure the end date is not included in the rebalance dates
        self.rebalance_dates = self.rebalance_dates[self.rebalance_dates != end_date]

        self.validate_data()


        self.portfolio = pd.Series(index=self.input_tickers,data=0.0,name=self.port_name)
        self.portfolio['Cash'] = initial_capital
        
        # Master dataframe to store the historical portfolio holdings
        self.portfolio_history_df = pd.DataFrame(index=self.strat_dates,columns=self.input_tickers)
        # self.portfolio_history_df.index.name

        # Just a catch all for any additional parameters that may be passed in for a substrategy
        self.params = params
    
    def validate_data(self) -> None:

        # Check that the input tickers are in the data blob
        for ticker in self.input_tickers:
            if ticker not in self.data_blob.tickers:
                raise ValueError(f'Ticker {ticker} not in data blob. Please check the input tickers.')

        # Check that the input weights sum to 1
        if np.abs(np.sum(self.input_weights) - 1) > 1e-8:
            raise ValueError('Input weights do not sum to 1. Please check the input weights.')

        

    def __repr__(self) -> str:
        return f'{self.short_name}: {self.start_date} - {self.end_date}'

    @property
    def port_value(self) -> float:
        return self.portfolio.sum()

    def rebalance_to_target_weights(self,target_weights:pd.Series) -> None:
        '''Rebalance the portfolio to the target weights provided. This will implictily involve selling off any
          securities that are overweight and buying any securities that are underweight.
    
        '''
    
        # Multiply the target weights by the current portfolio value to get the target value for each security
        target_values = target_weights * self.port_value

        # Update the new portfolio with the target values 
        # (This is implicitly carrying out trades...)
        self.portfolio = target_values
        self.portfolio_history_df.loc[self.current_date] = self.portfolio
    

    def increment_portfolio_by_returns(self) -> None:
        '''Increase the portfolio value by the returns for the current date. '''

        # If there is a return for the current date, then increment the portfolio by the returns
        if self.current_date in self.rets_df.index:
            self.portfolio = self.portfolio * (1 + self.rets_df.loc[self.current_date])
            
        # Regardless of if portfolio was incremented up or not, store the current portfolio value in the history for
        #  today's date. So we always have an estimated value for the portfolio at the end of each day.
        self.portfolio_history_df.loc[self.current_date] = self.portfolio

    
    def get_target_weights(self) -> pd.Series:
        '''Get the target weights for the portfolio based on the input weights. '''

        # Create a series with the input weights and the cash weight
        target_weights = pd.Series(index=self.input_tickers,data=self.input_weights)

        return target_weights

    def run_backtest(self,verbose=False) -> None:

        # Allocate the initial capital to the target weights
        target_weights = self.get_target_weights()
        self.rebalance_to_target_weights(target_weights)

        # Iterate through all the dates in the chosen time period
        for date in self.strat_dates[1:]:
            
            # Update the current date
            self.current_date = date

            # Increment the portfolio by the returns for the current date
            self.increment_portfolio_by_returns()

            # If the current date is a rebalance date, then rebalance the portfolio
            if date in self.rebalance_dates:
                if verbose:
                    print(f'Current Time {datetime.datetime.now()} Rebalancing: {date}')
                target_weights = self.get_target_weights()
                self.rebalance_to_target_weights(target_weights)

        # Calculate some useful data based on the portfolio history
        self.calculate_data()


    def calculate_data(self) -> None:
        '''Calculate some useful data based on the portfolio history which is nice to have when analyzing results.'''
        
        self.total_port_values = self.portfolio_history_df.sum(axis=1).astype(float).rename(self.port_name)
        self.weights_df = (self.portfolio_history_df.div(self.total_port_values,axis=0)).astype(float)
        self.wealth_index = self.total_port_values / self.total_port_values.iloc[0]

        self.cumulative_port_returns = self.wealth_index - 1

        # Calculate portfolio returns based on the total portfolio values
        self.portfolio_returns_all = self.total_port_values.pct_change().dropna()
        # self.portfolio_returns_all.name = self.short_name

        # But also calculate a portfolio return that removes days where the portfolio return is 0, because those
        # are with 99.999% certainty just holidays.
        basically_zero_mask = np.abs(self.portfolio_returns_all - 0) < 1e-8
        self.port_returns = self.portfolio_returns_all[~basically_zero_mask].copy()

        ''



if __name__ == '__main__':
    data = dd.DataEngine.load_saved_data() 
    bt = Backtester(data_blob=data,tickers=['AAPL','MSFT'],weights=[0.5,0.5],start_date='2010-01-01',end_date='2020-01-01')
    bt.run_backtest()
    print(bt.portfolio_history_df)

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\data_engine.py

```py
import os
import time
import pandas as pd
import yfinance as yf
import datetime as dt
import reference.constants as C
import streamlit as st

# DATA_FOLDER = 'data/'
DATA_FOLDER = 'temp_data/'
CACHE_EXPIRATION = 28800  # 8ish hours
MAX_FILES_SAVED = 100

class DataEngine:
    def __init__(self) -> None:
        self.adjusted_prices_df: pd.DataFrame = None
        self.rets_df: pd.DataFrame = None
        self.price_df: pd.DataFrame = None
        self.raw_data_df: pd.DataFrame = None

    def is_cache_expired(self, ticker: str) -> bool:
        file_path = f'{DATA_FOLDER}{ticker}.csv'
        if not os.path.exists(file_path):
            return True
        return (time.time() - os.path.getmtime(file_path)) > CACHE_EXPIRATION 

    def load_local_data(self, tickers: list[str]) -> pd.DataFrame:
        """Load data from local storage if available and not expired"""

        dfs = []
        for ticker in tickers:
            if self.is_cache_expired(ticker):
                # If any of the tickers are expired, return None (Meaning we will re-download everything)
                return None
            file_path = f'{DATA_FOLDER}{ticker}.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
                dfs.append(df)
        return pd.concat(dfs, axis=1) if dfs else None

    def check_storage_limit(self):
        """Checks if adding new files will exceed MAX_FILES_SAVED and clears folder if necessary"""
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)
            return
        
        existing_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
        if len(existing_files) >= MAX_FILES_SAVED:
            print("Storage limit exceeded, clearing folder...")
            for file in existing_files:
                os.remove(os.path.join(DATA_FOLDER, file))

    def save_data_locally(self, df: pd.DataFrame, tickers: list[str]) -> None:
        """Save downloaded data locally after checking storage limit"""
        self.check_storage_limit()
        os.makedirs(DATA_FOLDER, exist_ok=True)
        for ticker in tickers:
            mini_df = df[[ticker]].droplevel(0, axis=1).dropna()
            mini_df.to_csv(f'{DATA_FOLDER}{ticker}.csv')

    def download_new_data(self, tickers: list[str]) -> pd.DataFrame:
        tickers = list(dict.fromkeys(tickers))  # Remove duplicates
        # local_data = self.load_local_data(tickers)
        # if local_data is not None and not local_data.isnull().all().all():
        #     # print("Using cached data")
        #     self.raw_data_df = local_data
        # else:
        #     # print("Fetching new data from Yahoo Finance")
        self.raw_data_df = yf.download(tickers, group_by='ticker', auto_adjust=False, actions=False)
        self.save_data_locally(self.raw_data_df, tickers)

        self.clean_data()
        return self.rets_df

    def clean_data(self) -> pd.DataFrame:
        df = self.raw_data_df.copy()
        df.index = pd.to_datetime(df.index)
        
        self.price_df = df.loc[:, (slice(None), 'Close')]
        self.price_df.columns = self.price_df.columns.droplevel(1)

        adjusted_prices_df = df.loc[:, (slice(None), 'Adj Close')].copy()
        adjusted_prices_df.columns = adjusted_prices_df.columns.droplevel(1)
        adjusted_prices_df.ffill(inplace=True)
        
        self.rets_df = adjusted_prices_df.pct_change(fill_method=None)
        self.rets_df = self.rets_df[sorted(self.rets_df.columns)].copy()
        self.adjusted_prices_df = adjusted_prices_df
        
        return self.rets_df

    def save_data(self, folder_path=DATA_FOLDER) -> None:
        os.makedirs(folder_path, exist_ok=True)
        self.rets_df.to_csv(f'{folder_path}rets_df.csv')
        self.adjusted_prices_df.to_csv(f'{folder_path}adjusted_prices_df.csv')
        self.price_df.to_csv(f'{folder_path}price_df.csv')

    @staticmethod
    def load_saved_data(folder: str = DATA_FOLDER) -> "DataEngine":
        dblob = DataEngine()
        dblob.rets_df = pd.read_csv(f'{folder}rets_df.csv', index_col=0, parse_dates=True)
        dblob.adjusted_prices_df = pd.read_csv(f'{folder}adjusted_prices_df.csv', index_col=0, parse_dates=True)
        dblob.price_df = pd.read_csv(f'{folder}price_df.csv', index_col=0, parse_dates=True)
        dblob.price_df.index = pd.to_datetime(dblob.price_df.index)
        return dblob
    
    @property
    def tickers(self) -> list[str]:
        return self.rets_df.columns.tolist()

if __name__ == '__main__':
    downloader = DataEngine()
    downloader.download_new_data(C.MARKET_TICKERS)
    downloader.load_local_data(C.MARKET_TICKERS)
    print('Done!')

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\home.py

```py
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as dt

import inputs as inputs
import data_engine as dd
import backtester as bt
import results as rs

st.title("Portfolio Backtester")
html_title = """
<h4 style="font-family: 'Arial', sans-serif; font-size: 1.2rem; font-weight: bold; font-style: italic; color: #555; margin-top: -10px; margin-bottom: 20px; text-shadow: 1px 1px #ddd;">A Brandt Green Production</h4>
"""
st.markdown(html_title, unsafe_allow_html=True)

st.sidebar.markdown("## Table of Contents")
st.sidebar.markdown("""
- [Inputs](#inputs)
- [Results](#results)
  - [Cumulative Returns](#cumulative-returns)
  - [Volatility](#volatility)    
  - [Performance Metrics](#performance-metrics)                
  - [Correlation Matrix](#correlation-matrix)                                        
  - [Portfolio Weights Over Time](#portfolio-weights-over-time)
  - [Individual Prices](#individual-prices)
- [Raw Data Reference](#raw-data-reference)
  - [Rebalance Dates](#rebalance-dates)                    
  - [Raw Returns](#raw-returns)
  - [Raw Port Weights](#raw-portfolio-weights)
""", unsafe_allow_html=True)




# ----------------------------
# Collect User Inputs
# ----------------------------

cleaned_inputs = inputs.get_user_inputs()

# run_backtest = st.button("Run Backtest")


# ----------------------------
# Fetch Market Data & Validate against Inputs
# ----------------------------

needed_tickers = list(dict.fromkeys(cleaned_inputs.tickers + [cleaned_inputs.bench_ticker]))


# Need to uncomment out below in a bit
# if not run_backtest:
#     st.stop()


with st.spinner("Fetching data..."):
    data = dd.DataEngine()
    
    # Try loading cached data first
    data.raw_data_df = data.load_local_data(needed_tickers)
    
    # May need to review below to fetch data for any new tickers
    if data.raw_data_df is None or cleaned_inputs.fetch_new_data:
        # st.warning("Fetching new data from Yahoo Finance. This may take a second...")
        with st.spinner("Fetching new data from Yahoo Finance. This may take a second..."):
            data.download_new_data(needed_tickers)
    
    # Make sure it's been cleaned
    data.clean_data()


# Validate we have the data to run a backtest
# Ensure selected tickers exist in dataset (Should be moved somewhere else???)
missing_tickers = [t for t in cleaned_inputs.tickers if t not in data.tickers]
if missing_tickers:
    error_msg = f"""Missing data for some tickers. Sorry... If you want to fetch new data, toggle the buttom
\n Missing tickers: {missing_tickers}"""
    st.error(error_msg)
    st.stop()


# Filter returns dataframe for only the selected tickers
data.rets_df = data.rets_df[needed_tickers].copy()



# Check that we have returns for all tickers for the entire backtest period
missing_returns = data.rets_df.loc[cleaned_inputs.start_date:cleaned_inputs.end_date].isnull().sum()
if missing_returns.any():
    error_msg = f"""Missing returns for some tickers during the backtest period. Sorry... 
\n Problem tickers: {missing_returns[missing_returns > 0].index.tolist()}"""
    st.error(error_msg)
    st.stop()



# ----------------------------
# Run Backtest
# ----------------------------

with st.spinner("Running backtest..."):
    backtester = bt.Backtester(
        data_blob=data,
        tickers=cleaned_inputs.tickers,
        weights=cleaned_inputs.weights,
        start_date=str(cleaned_inputs.start_date),
        end_date=str(cleaned_inputs.end_date),
        rebal_freq=cleaned_inputs.rebalance_freq,
        port_name=cleaned_inputs.port_name,
    )
    backtester.run_backtest()

# ----------------------------
# Display Results
# ----------------------------

# Add a few line breaks and a separator to distinguish the results section
st.markdown("---")
st.markdown("## Results")

rs.display_results(backtester, data, cleaned_inputs)

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\inputs.py

```py
from dataclasses import dataclass
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as dt
import reference.constants as C
from 
from utils import DynamicDates

# Add a dataclass to hold the user inputs


@dataclass
class CleanInputs:

    tickers: list
    weights: str
    start_date: dt.datetime
    end_date: dt.datetime
    port_name: str
    rebalance_freq: str
    bench_ticker: str
    fetch_new_data: bool = False



def get_user_inputs():
    """Collects user inputs and returns them as variables."""

    st.markdown("## Inputs")

    # ------------------
    # Ticker Input
    # ------------------

    st.markdown("#### Tickers")
    default_ticks = ' '.join(C.SECTOR_TICKERS)
    tickers_input = st.text_area(
        "Enter tickers separated by spaces (e.g., AAPL MSFT AMZN GOOGL META TSLA JPM):",
        default_ticks
    )

    tickers = [t.strip().upper() for t in tickers_input.split(" ") if t.strip()]

    TICKER_LIMIT = 50
    if len(tickers) > TICKER_LIMIT:
        st.error(f'Sorry, for now the maximum tickers allowed is {TICKER_LIMIT}. Because I am worried about abusing the API. ')
        st.stop()

    # Raise error if there are duplicates
    if len(tickers) != len(set(tickers)):
        dups = set([ticker for ticker in tickers if tickers.count(ticker) > 1])
        st.error(f"Duplicate tickers found. Please remove: {dups}")

    # ------------------
    # Weights Input
    # ------------------
    st.markdown("#### Portfolio Weights")

    equal_weights = [1 / len(tickers) * 100]  * len(tickers)
    equal_weights = [f'{round(w, 2)}' for w in equal_weights]
    equal_weights_str = " ".join(equal_weights)
    weights_msg = "Enter the target weights for each ticker. Defaults to equal-weight. Space-separated. Percentagses. Should sum to 1, e.g, 35 25 40:"
    weights_input = st.text_area(
        weights_msg,
        equal_weights_str
    )

    weights_input = weights_input.split(" ")
    # Make sure the number of weights matches the number of tickers
    if len(weights_input) != len(tickers):
        st.error(f"Number of weights does not match number of tickers. Please provide a weight for each ticker.")
        st.stop()

    # Convert to floats
    weights_input = [float(w)/100 for w in weights_input]

    # Validat that weights are closish to 1

    DIFF_THRESHOLD = .05
    if abs(1 - sum(weights_input)) > DIFF_THRESHOLD:

        st.error(f"Your weights do not sum to 1. Please ensure they sum to 1. Current sum: {sum(weights_input)}")
        st.stop()
    
    # Rescale the weights anyway (to handle when they are super close)
    weights_input = [w / sum(weights_input) for w in weights_input]

    # ------------------
    # Date Selection
    # ------------------

    st.markdown("#### Date Range")

    # Date range selection dropdown
    date_option = st.selectbox(
        "Select a lookback range (this just assists in picking start date):",
        ["Custom", "1D", "YTD", "1 Year", "3 Years", "5 Years", "10 Years", "15 Years"],
        index=2,  # Default to "YTD"
    )

    # Automatically update start date based on selection
    date_dict = {
        "1D": DynamicDates.day_before_yesterday(),
        "YTD": DynamicDates.prior_year_end(),
        "1 Year": DynamicDates.one_year_ago(),
        "3 Years": DynamicDates.three_years_ago(),
        "5 Years": DynamicDates.five_years_ago(),
        "10 Years": DynamicDates.ten_years_ago(),
        "15 Years": DynamicDates.fifteen_years_ago(),
    }


    start_date_default = date_dict.get(date_option,None)
    if start_date_default is None:
        # If the default is none, 
        start_date_default = DynamicDates.day_before_yesterday()

    # Start date (users can override)
    start_date = st.date_input("Start Date (assumes you invest at close of this date):", start_date_default)

    # End Date Selection
    # if date_option != 'Custom':
    end_date = st.date_input("End Date (assumes you liquidate at close of this date):", DynamicDates.yesterday())
    # else:
    #     end_date = st.date_input("End Date (Assumes you liquidate at close of this date):")
    # ------------------
    # Rebalance Frequency Selection
    # ------------------

    st.markdown("#### Rebalancing Options")
    rebalance_freq = st.selectbox(
        "Select rebalance frequency:",
        [
            "YE - Year end",
            "YS - Year start",
            "QS - Quarter start",
            "QE - Quarter end",
            "MS - Month start",
            "ME - Month end",
            "W - Weekly",
            "D - Calendar day",
        ],
        index=3  # Default to "QE - Quarter end"
    )

    rebalance_freq = rebalance_freq.split(" - ")[0]  # Extract alias


    # ------------------
    # Benchmark Selection
    # ------------------
    st.markdown("#### Benchmark")
    benchmark = st.selectbox(
        "Select a benchmark (only used to calculate beta):",
        ["SPY", "IWM","QQQ","BND"],
        index=0
    )



    # ------------------
    # Portfolio Name
    # ----------------
    port_name = st.text_input("Enter a name for your portfolio:", "Port")
    # st.markdown('---')

    # Add a toggle to fetch new date or use old
    # st.markdown("#### Force Fetch New Data")
    fetch_new_data = False
    # st.write("If you need to fetch new data (not using data that is cached), toggle the switch below. You should probably never need to do this.")
    # fetch_new_data = st.toggle("Query Updated Data", value=False)

    
    
    clean_inputs = CleanInputs(
        tickers=tickers,
        weights=weights_input,
        start_date=start_date,
        end_date=end_date,
        port_name=port_name,
        rebalance_freq=rebalance_freq,
        bench_ticker=benchmark,
        fetch_new_data=fetch_new_data
    )
    return clean_inputs
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\metrics.py

```py
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

def calculate_beta(rets:pd.Series,bench_rets:pd.Series) -> float:
    '''Calculate the beta of the portfolio returns against the benchmark returns.'''
    
    # Join series together to have matching dates
    both_rets = pd.concat([rets,bench_rets],axis=1).dropna()

    cov_matrix = both_rets.cov()
    beta = cov_matrix.iloc[0,1] / cov_matrix.iloc[1,1]

    return beta

def calculate_alpha(returns:pd.Series,bench_rets:pd.Series) -> float:
    '''Calculate annualzied alpha.'''

    # Create a cleaneddataframe so we can feed it into statsmodels
    data = pd.concat([returns,bench_rets],axis=1).dropna()
    data.columns = ['port','bench']
    model = smf.ols('port ~ bench',data=data).fit()
    alpha = model.params['Intercept'] * 252

    return alpha


def upside_capture(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    '''Calculate the upside capture ratio using average returns.'''
    
    # Join series together to align dates
    both_rets = pd.concat([port_rets, bench_rets], axis=1).dropna()
    up_market_rets = both_rets[both_rets.iloc[:, 1] > 0]  # Use only periods where the benchmark is positive

    # Calculate average returns
    port_avg_ret = up_market_rets.iloc[:, 0].mean()
    bench_avg_ret = up_market_rets.iloc[:, 1].mean()

    # Avoid division by zero
    if bench_avg_ret == 0:
        return np.nan

    # Calculate upside capture ratio
    up_capture = port_avg_ret / bench_avg_ret
    return up_capture


def downside_capture(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    '''Calculate the downside capture ratio using average returns.'''
    
    # Join series together to align dates
    both_rets = pd.concat([port_rets, bench_rets], axis=1).dropna()
    down_market_rets = both_rets[both_rets.iloc[:, 1] < 0]  # Use only periods where the benchmark is negative

    # Calculate average returns
    port_avg_ret = down_market_rets.iloc[:, 0].mean()
    bench_avg_ret = down_market_rets.iloc[:, 1].mean()

    # Avoid division by zero
    if bench_avg_ret == 0:
        return np.nan

    # Calculate downside capture ratio
    down_capture = port_avg_ret / bench_avg_ret
    return down_capture


def get_downside_deviation(returns, target=0):
    downside_diff = np.maximum(0, target - returns)
    squared_diff = np.square(downside_diff)
    mean_squared_diff = np.nanmean(squared_diff)
    dd = np.sqrt(mean_squared_diff) * np.sqrt(252)
    return dd


def get_max_drawdown(returns:pd.Series) -> float:

    wealth_index = (1 + returns).cumprod().array

    # Insert a wealth index of 1 at the beginning to make the calculation work
    wealth_index = np.insert(wealth_index,0,1)
    # Get the cumulative max
    cum_max = np.maximum.accumulate(wealth_index)
    max_dd = ((wealth_index / cum_max) - 1).min()

    return max_dd


def calculate_metrics(returns:pd.Series,bench_rets:pd.Series) -> dict:
    '''Calculate the key metrics for a given series of returns. Assumes returns are daily.'''

    total_ret = (1+returns).prod() - 1
    cagr = (total_ret + 1) ** (252 / returns.shape[0]) - 1
    vol = returns.std() * np.sqrt(252)
    max_dd = get_max_drawdown(returns)

    # Sharpe Ratio
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    alpha = calculate_alpha(returns,bench_rets)


    beta = calculate_beta(returns,bench_rets)
    downside_deviation = get_downside_deviation(returns)

    up_capture = upside_capture(returns,bench_rets)
    down_capture = downside_capture(returns,bench_rets)

    metrics = {
        'Total Return': total_ret,
        'CAGR': cagr,
        'Volatility': vol,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Beta': beta,
        'Alpha': alpha,
        'Downside Deviation': downside_deviation,
        'Up Capture': up_capture,
        'Down Capture': down_capture

    }

    return pd.Series(metrics)

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\requirements.txt

```txt
pandas
numpy
datetime
yfinance
streamlit
plotly
statsmodels
matplotlib

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\results.py

```py
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as dt

import streamlit_app.data_engine as dd
import backtester as bt
import streamlit_app.inputs as inputs
import metrics
import utils


# Utility Functions
def format_as_percent(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Format specified columns as percentages."""
    for column in columns:
        df[column] = df[column].map('{:.2%}'.format)
    return df


def plot_line_chart(df: pd.DataFrame, title: str, yaxis_title: str) -> None:
    fig = px.line(df, title=title)
    fig.update_yaxes(tickformat=".2%", title_text=yaxis_title)
    fig.update_xaxes(title_text="Date")
    st.plotly_chart(fig)


def plot_bar_chart(df: pd.Series, title: str, yaxis_title: str) -> None:
    fig = px.bar(df, title=title)
    fig.update_yaxes(tickformat=".2%", title_text=yaxis_title)
    fig.update_xaxes(title_text="")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)


def display_results(backtest:bt.Backtester,data:dd.DataEngine, cleaned_inputs:inputs.CleanInputs) -> None:

    start_dt = pd.to_datetime(cleaned_inputs.start_date)

    # Security returns should be after the start date (non inclusive) and before the end date (inclusive)
    rets_filered_df = data.rets_df[data.rets_df.index > start_dt].loc[:cleaned_inputs.end_date]
    security_rets_df = rets_filered_df[cleaned_inputs.tickers]
    bench_rets = rets_filered_df[cleaned_inputs.bench_ticker]
    all_rets_df = pd.concat([backtest.port_returns, security_rets_df], axis=1)
    security_prices_df = data.price_df[data.price_df.index > start_dt].loc[:cleaned_inputs.end_date]
    security_prices_df = security_prices_df[cleaned_inputs.tickers]

    bench_rets = rets_filered_df[cleaned_inputs.bench_ticker]

    st.markdown("### Cumulative Returns")

    cum_rets_df = (1 + all_rets_df).cumprod() - 1

    plot_line_chart(cum_rets_df, "Cumulative Returns", "Cumulative Returns")
    
    # Bar plot of total return
    total_rets = cum_rets_df.iloc[-1].sort_values(ascending=False)
    plot_bar_chart(total_rets, "Total Returns", "Total Return")

    # Add on yearly returns, if we have enough data
    annual_rets = all_rets_df.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    if len(annual_rets) > 1:
        st.markdown("#### Annual Returns")        
        annual_rets.index = annual_rets.index.year
        annual_rets = annual_rets.T
        # Format these returns as a heatmap each year
        annual_rets = annual_rets.style.format("{:.2%}").background_gradient(cmap='RdYlGn', axis=1)


        st.write(annual_rets)

    # Volatility
    # Display the vol in a bar chart in the same order as the total rets
    st.markdown("### Volatility")    
    total_vol = all_rets_df.std() * 252 ** 0.5

    total_vol = total_vol[total_rets.index]
    plot_bar_chart(total_vol, "Total Period Annualized Volatility", "Volatility")

    # If you have enough data, plot the rolling vol
    ROLLING_WINDOW = 252
    if len(all_rets_df) > ROLLING_WINDOW:
    
        rolling_vols = all_rets_df.rolling(window=252).std() * 252 ** 0.5
        rolling_vols = rolling_vols.dropna()
        plot_line_chart(rolling_vols, "Rolling 1-Year Volatility", "Volatility")

    #---------------------------
    # Metrics
    #---------------------------

    st.markdown("### Performance Metrics")    
    metrics_df = all_rets_df.apply(metrics.calculate_metrics, args=(bench_rets,),axis=0)
    # We want to apply lots of fun formatting to the metrics
    metrics_pretty_df = metrics_df.T.copy()
    COLS_TO_PRETTIFY = ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown', 'Alpha', 'Downside Deviation']
    metrics_pretty_df = format_as_percent(metrics_pretty_df, COLS_TO_PRETTIFY)
    # Format the following columns to onlu 2 decimal places
    DECIMAL_COLS = ['Beta', 'Sharpe', 'Up Capture', 'Down Capture']
    metrics_pretty_df[DECIMAL_COLS] = metrics_pretty_df[DECIMAL_COLS].map('{:.2f}'.format)

    st.write(metrics_pretty_df)

    # ----------------------------
    # Correlation Matrix
    # ----------------------------
    st.markdown("### Correlation Matrix")
    corr = all_rets_df.corr()
    corr_pretty_df = corr.copy()
    # # corr_pretty_df = corr_pretty_df.applymap('{:.2f}'.format)
    corr_pretty_df = corr.style.format("{:.2f}").background_gradient(cmap='coolwarm', vmin=-1, vmax=1)
    st.write(corr_pretty_df)


    # Portfolio Weights Over Time
    st.markdown("### Portfolio Weights Over Time")
    plot_line_chart(backtest.weights_df, "Portfolio Weights Over Time", "Weight")


    # # Returns
    # st.markdown("### Individual Returns")
    # # Add a tab for each of the securities and show a plot of it's indivdiual cumualtive return
    # ret_tabs = st.tabs(cum_rets_df.columns.to_list())
    # for ticker, tab in zip(cum_rets_df.columns, ret_tabs):
    #     with tab:
    #         total_ret = cum_rets_df[ticker].iloc[-1]
    #         st.write(f"Total Return: {total_ret:.2%}")
    #         plot_line_chart(cum_rets_df[ticker], f"{ticker} Cumulative Return", "Cumulative Return")

    #         # Add on another bar chart that shows the return on different time periods.


    st.markdown("### Individual Prices")
    st.write('_These prices are not adjusted for splits or dividends_')
    price_tabs = st.tabs(security_prices_df.columns.to_list())
    for ticker, tab in zip(security_prices_df.columns, price_tabs):
        with tab:
            fig = px.line(security_prices_df[ticker], title=f"{ticker} Prices")
            # Format in dollars
            fig.update_yaxes(title_text="Price", tickprefix="$")
            fig.update_xaxes(title_text="Date")
            st.plotly_chart(fig)


    

    # # ----------------------------
    # # Display Raw Data
    # # ----------------------------

    st.markdown("## Raw Data Reference")

    st.markdown("### Rebalance Dates")
    dates = backtest.rebalance_dates
    dates.name = 'Rebalance Dates'
    dates = pd.Series(dates.date, name='Rebalance Dates')
    st.write(dates)



    st.markdown("### Raw Returns")
    rets_df = utils.convert_dt_index(all_rets_df)
    # Format the returns as percentages and color code them. Positive returns are green, negative are red.
    styled_df = rets_df.style.format("{:.2%}").map(utils.color_returns)
    st.write(styled_df)

    # st.markdown("### Portfolio History")
    # st.write(utils.convert_dt_index(backtest.portfolio_history_df))

    st.markdown("### Raw Portfolio Weights")
    weights_df = utils.convert_dt_index(backtest.weights_df)
    weights_df = weights_df.map('{:.2%}'.format)
    st.write(weights_df)


if __name__ == '__main__':

    data = dd.DataEngine.load_saved_data() 
    back = bt.Backtester(data_blob=data,tickers=['AAPL','MSFT'],weights=[0.5,0.5],start_date='2010-01-01',end_date='2020-01-01')

    back.run_backtest()


    cleaned_inputs = inputs.CleanInputs(tickers=['AAPL','MSFT'],weights='0.5,0.5',start_date=dt.datetime(2010,1,1),end_date=dt.datetime(2020,1,1),port_name='Portfolio',rebalance_freq='QE',fetch_new_data=False,bench_ticker='SPY')

    display_results(back,data,cleaned_inputs)
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\utils.py

```py
import pandas as pd
import datetime as dt

def convert_dt_index(df:pd.DataFrame) -> pd.DataFrame:
    '''Convert the index of a DataFrame from datetime to date'''
    df.index = pd.to_datetime(df.index).date
    return df

def color_returns(val):
    color = "green" if val > 0 else "red"
    return f"color: {color}"


class DynamicDates:
    @classmethod
    def today(cls):
        return dt.datetime.today()

    @classmethod
    def yesterday(cls):
        return cls.today() - dt.timedelta(days=1)

    @classmethod
    def day_before_yesterday(cls):
        return cls.today() - dt.timedelta(days=2)

    @classmethod
    def prior_year_end(cls):
        return dt.datetime(cls.today().year - 1, 12, 31)

    @classmethod
    def one_year_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 1)

    @classmethod
    def three_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 3)

    @classmethod
    def five_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 5)

    @classmethod
    def ten_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 10)

    @classmethod
    def fifteen_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 15)


if __name__ == '__main__':
    print("Today:", DynamicDates.today())
    print("Yesterday:", DynamicDates.yesterday())
    print("One Year Ago:", DynamicDates.one_year_ago())

```


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

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\backtester.py

```py
import pandas as pd
import datetime
import numpy as np
import streamlit_app.data_engine as dd


# General, helper variables that are used later. 
all_dates = pd.date_range(start='1900-01-01',end='2099-12-31') # Big date range to cover all reasonable dates we may include in our backtest (This is filtered later)


class Backtester:

    pretty_name = 'BaseStrategy'
    short_name = 'BaseStrat'

    def __init__(
        self,
        data_blob: dd.DataEngine,
        tickers: list[str],
        weights: list[float],
        start_date: str,
        end_date: str,
        initial_capital: float = 1_000_000,
        rebal_freq: str = 'QE',
        port_name: str = 'Port',
        params: dict = {}
    ) -> None:

        self.data_blob = data_blob
        self.rets_df = data_blob.rets_df
        self.input_tickers = tickers
        self.input_weights = weights
        self.port_name = port_name
        self.start_date = start_date
        self.end_date = end_date
        self.current_date = start_date
        self.strat_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)] 
        self.rebalance_dates = pd.date_range(start=start_date,end=end_date,freq=rebal_freq) 
        # Make sure the end date is not included in the rebalance dates
        self.rebalance_dates = self.rebalance_dates[self.rebalance_dates != end_date]

        self.validate_data()


        self.portfolio = pd.Series(index=self.input_tickers,data=0.0,name=self.port_name)
        self.portfolio['Cash'] = initial_capital
        
        # Master dataframe to store the historical portfolio holdings
        self.portfolio_history_df = pd.DataFrame(index=self.strat_dates,columns=self.input_tickers)
        # self.portfolio_history_df.index.name

        # Just a catch all for any additional parameters that may be passed in for a substrategy
        self.params = params
    
    def validate_data(self) -> None:

        # Check that the input tickers are in the data blob
        for ticker in self.input_tickers:
            if ticker not in self.data_blob.tickers:
                raise ValueError(f'Ticker {ticker} not in data blob. Please check the input tickers.')

        # Check that the input weights sum to 1
        if np.abs(np.sum(self.input_weights) - 1) > 1e-8:
            raise ValueError('Input weights do not sum to 1. Please check the input weights.')

        

    def __repr__(self) -> str:
        return f'{self.short_name}: {self.start_date} - {self.end_date}'

    @property
    def port_value(self) -> float:
        return self.portfolio.sum()

    def rebalance_to_target_weights(self,target_weights:pd.Series) -> None:
        '''Rebalance the portfolio to the target weights provided. This will implictily involve selling off any
          securities that are overweight and buying any securities that are underweight.
    
        '''
    
        # Multiply the target weights by the current portfolio value to get the target value for each security
        target_values = target_weights * self.port_value

        # Update the new portfolio with the target values 
        # (This is implicitly carrying out trades...)
        self.portfolio = target_values
        self.portfolio_history_df.loc[self.current_date] = self.portfolio
    

    def increment_portfolio_by_returns(self) -> None:
        '''Increase the portfolio value by the returns for the current date. '''

        # If there is a return for the current date, then increment the portfolio by the returns
        if self.current_date in self.rets_df.index:
            self.portfolio = self.portfolio * (1 + self.rets_df.loc[self.current_date])
            
        # Regardless of if portfolio was incremented up or not, store the current portfolio value in the history for
        #  today's date. So we always have an estimated value for the portfolio at the end of each day.
        self.portfolio_history_df.loc[self.current_date] = self.portfolio

    
    def get_target_weights(self) -> pd.Series:
        '''Get the target weights for the portfolio based on the input weights. '''

        # Create a series with the input weights and the cash weight
        target_weights = pd.Series(index=self.input_tickers,data=self.input_weights)

        return target_weights

    def run_backtest(self,verbose=False) -> None:

        # Allocate the initial capital to the target weights
        target_weights = self.get_target_weights()
        self.rebalance_to_target_weights(target_weights)

        # Iterate through all the dates in the chosen time period
        for date in self.strat_dates[1:]:
            
            # Update the current date
            self.current_date = date

            # Increment the portfolio by the returns for the current date
            self.increment_portfolio_by_returns()

            # If the current date is a rebalance date, then rebalance the portfolio
            if date in self.rebalance_dates:
                if verbose:
                    print(f'Current Time {datetime.datetime.now()} Rebalancing: {date}')
                target_weights = self.get_target_weights()
                self.rebalance_to_target_weights(target_weights)

        # Calculate some useful data based on the portfolio history
        self.calculate_data()


    def calculate_data(self) -> None:
        '''Calculate some useful data based on the portfolio history which is nice to have when analyzing results.'''
        
        self.total_port_values = self.portfolio_history_df.sum(axis=1).astype(float).rename(self.port_name)
        self.weights_df = (self.portfolio_history_df.div(self.total_port_values,axis=0)).astype(float)
        self.wealth_index = self.total_port_values / self.total_port_values.iloc[0]

        self.cumulative_port_returns = self.wealth_index - 1

        # Calculate portfolio returns based on the total portfolio values
        self.portfolio_returns_all = self.total_port_values.pct_change().dropna()
        # self.portfolio_returns_all.name = self.short_name

        # But also calculate a portfolio return that removes days where the portfolio return is 0, because those
        # are with 99.999% certainty just holidays.
        basically_zero_mask = np.abs(self.portfolio_returns_all - 0) < 1e-8
        self.port_returns = self.portfolio_returns_all[~basically_zero_mask].copy()

        ''



if __name__ == '__main__':
    data = dd.DataEngine.load_saved_data() 
    bt = Backtester(data_blob=data,tickers=['AAPL','MSFT'],weights=[0.5,0.5],start_date='2010-01-01',end_date='2020-01-01')
    bt.run_backtest()
    print(bt.portfolio_history_df)

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\data_engine.py

```py
import os
import time
import pandas as pd
import yfinance as yf
import datetime as dt
import reference.constants as C
import streamlit as st

# DATA_FOLDER = 'data/'
DATA_FOLDER = 'temp_data/'
CACHE_EXPIRATION = 28800  # 8ish hours
MAX_FILES_SAVED = 100

class DataEngine:
    def __init__(self) -> None:
        self.adjusted_prices_df: pd.DataFrame = None
        self.rets_df: pd.DataFrame = None
        self.price_df: pd.DataFrame = None
        self.raw_data_df: pd.DataFrame = None

    def is_cache_expired(self, ticker: str) -> bool:
        file_path = f'{DATA_FOLDER}{ticker}.csv'
        if not os.path.exists(file_path):
            return True
        return (time.time() - os.path.getmtime(file_path)) > CACHE_EXPIRATION 

    def load_local_data(self, tickers: list[str]) -> pd.DataFrame:
        """Load data from local storage if available and not expired"""

        dfs = []
        for ticker in tickers:
            if self.is_cache_expired(ticker):
                # If any of the tickers are expired, return None (Meaning we will re-download everything)
                return None
            file_path = f'{DATA_FOLDER}{ticker}.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
                dfs.append(df)
        return pd.concat(dfs, axis=1) if dfs else None

    def check_storage_limit(self):
        """Checks if adding new files will exceed MAX_FILES_SAVED and clears folder if necessary"""
        if not os.path.exists(DATA_FOLDER):
            os.makedirs(DATA_FOLDER)
            return
        
        existing_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
        if len(existing_files) >= MAX_FILES_SAVED:
            print("Storage limit exceeded, clearing folder...")
            for file in existing_files:
                os.remove(os.path.join(DATA_FOLDER, file))

    def save_data_locally(self, df: pd.DataFrame, tickers: list[str]) -> None:
        """Save downloaded data locally after checking storage limit"""
        self.check_storage_limit()
        os.makedirs(DATA_FOLDER, exist_ok=True)
        for ticker in tickers:
            mini_df = df[[ticker]].droplevel(0, axis=1).dropna()
            mini_df.to_csv(f'{DATA_FOLDER}{ticker}.csv')

    def download_new_data(self, tickers: list[str]) -> pd.DataFrame:
        tickers = list(dict.fromkeys(tickers))  # Remove duplicates
        # local_data = self.load_local_data(tickers)
        # if local_data is not None and not local_data.isnull().all().all():
        #     # print("Using cached data")
        #     self.raw_data_df = local_data
        # else:
        #     # print("Fetching new data from Yahoo Finance")
        self.raw_data_df = yf.download(tickers, group_by='ticker', auto_adjust=False, actions=False)
        self.save_data_locally(self.raw_data_df, tickers)

        self.clean_data()
        return self.rets_df

    def clean_data(self) -> pd.DataFrame:
        df = self.raw_data_df.copy()
        df.index = pd.to_datetime(df.index)
        
        self.price_df = df.loc[:, (slice(None), 'Close')]
        self.price_df.columns = self.price_df.columns.droplevel(1)

        adjusted_prices_df = df.loc[:, (slice(None), 'Adj Close')].copy()
        adjusted_prices_df.columns = adjusted_prices_df.columns.droplevel(1)
        adjusted_prices_df.ffill(inplace=True)
        
        self.rets_df = adjusted_prices_df.pct_change(fill_method=None)
        self.rets_df = self.rets_df[sorted(self.rets_df.columns)].copy()
        self.adjusted_prices_df = adjusted_prices_df
        
        return self.rets_df

    def save_data(self, folder_path=DATA_FOLDER) -> None:
        os.makedirs(folder_path, exist_ok=True)
        self.rets_df.to_csv(f'{folder_path}rets_df.csv')
        self.adjusted_prices_df.to_csv(f'{folder_path}adjusted_prices_df.csv')
        self.price_df.to_csv(f'{folder_path}price_df.csv')

    @staticmethod
    def load_saved_data(folder: str = DATA_FOLDER) -> "DataEngine":
        dblob = DataEngine()
        dblob.rets_df = pd.read_csv(f'{folder}rets_df.csv', index_col=0, parse_dates=True)
        dblob.adjusted_prices_df = pd.read_csv(f'{folder}adjusted_prices_df.csv', index_col=0, parse_dates=True)
        dblob.price_df = pd.read_csv(f'{folder}price_df.csv', index_col=0, parse_dates=True)
        dblob.price_df.index = pd.to_datetime(dblob.price_df.index)
        return dblob
    
    @property
    def tickers(self) -> list[str]:
        return self.rets_df.columns.tolist()

if __name__ == '__main__':
    downloader = DataEngine()
    downloader.download_new_data(C.MARKET_TICKERS)
    downloader.load_local_data(C.MARKET_TICKERS)
    print('Done!')

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\home.py

```py
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as dt

import inputs as inputs
import data_engine as dd
import backtester as bt
import results as rs

st.title("Portfolio Backtester")
html_title = """
<h4 style="font-family: 'Arial', sans-serif; font-size: 1.2rem; font-weight: bold; font-style: italic; color: #555; margin-top: -10px; margin-bottom: 20px; text-shadow: 1px 1px #ddd;">A Brandt Green Production</h4>
"""
st.markdown(html_title, unsafe_allow_html=True)

st.sidebar.markdown("## Table of Contents")
st.sidebar.markdown("""
- [Inputs](#inputs)
- [Results](#results)
  - [Cumulative Returns](#cumulative-returns)
  - [Volatility](#volatility)    
  - [Performance Metrics](#performance-metrics)                
  - [Correlation Matrix](#correlation-matrix)                                        
  - [Portfolio Weights Over Time](#portfolio-weights-over-time)
  - [Individual Prices](#individual-prices)
- [Raw Data Reference](#raw-data-reference)
  - [Rebalance Dates](#rebalance-dates)                    
  - [Raw Returns](#raw-returns)
  - [Raw Port Weights](#raw-portfolio-weights)
""", unsafe_allow_html=True)




# ----------------------------
# Collect User Inputs
# ----------------------------

cleaned_inputs = inputs.get_user_inputs()

# run_backtest = st.button("Run Backtest")


# ----------------------------
# Fetch Market Data & Validate against Inputs
# ----------------------------

needed_tickers = list(dict.fromkeys(cleaned_inputs.tickers + [cleaned_inputs.bench_ticker]))


# Need to uncomment out below in a bit
# if not run_backtest:
#     st.stop()


with st.spinner("Fetching data..."):
    data = dd.DataEngine()
    
    # Try loading cached data first
    data.raw_data_df = data.load_local_data(needed_tickers)
    
    # May need to review below to fetch data for any new tickers
    if data.raw_data_df is None or cleaned_inputs.fetch_new_data:
        # st.warning("Fetching new data from Yahoo Finance. This may take a second...")
        with st.spinner("Fetching new data from Yahoo Finance. This may take a second..."):
            data.download_new_data(needed_tickers)
    
    # Make sure it's been cleaned
    data.clean_data()


# Validate we have the data to run a backtest
# Ensure selected tickers exist in dataset (Should be moved somewhere else???)
missing_tickers = [t for t in cleaned_inputs.tickers if t not in data.tickers]
if missing_tickers:
    error_msg = f"""Missing data for some tickers. Sorry... If you want to fetch new data, toggle the buttom
\n Missing tickers: {missing_tickers}"""
    st.error(error_msg)
    st.stop()


# Filter returns dataframe for only the selected tickers
data.rets_df = data.rets_df[needed_tickers].copy()



# Check that we have returns for all tickers for the entire backtest period
missing_returns = data.rets_df.loc[cleaned_inputs.start_date:cleaned_inputs.end_date].isnull().sum()
if missing_returns.any():
    error_msg = f"""Missing returns for some tickers during the backtest period. Sorry... 
\n Problem tickers: {missing_returns[missing_returns > 0].index.tolist()}"""
    st.error(error_msg)
    st.stop()



# ----------------------------
# Run Backtest
# ----------------------------

with st.spinner("Running backtest..."):
    backtester = bt.Backtester(
        data_blob=data,
        tickers=cleaned_inputs.tickers,
        weights=cleaned_inputs.weights,
        start_date=str(cleaned_inputs.start_date),
        end_date=str(cleaned_inputs.end_date),
        rebal_freq=cleaned_inputs.rebalance_freq,
        port_name=cleaned_inputs.port_name,
    )
    backtester.run_backtest()

# ----------------------------
# Display Results
# ----------------------------

# Add a few line breaks and a separator to distinguish the results section
st.markdown("---")
st.markdown("## Results")

rs.display_results(backtester, data, cleaned_inputs)

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\inputs.py

```py
from dataclasses import dataclass
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as dt
import reference.constants as C
from 
from utils import DynamicDates

# Add a dataclass to hold the user inputs


@dataclass
class CleanInputs:

    tickers: list
    weights: str
    start_date: dt.datetime
    end_date: dt.datetime
    port_name: str
    rebalance_freq: str
    bench_ticker: str
    fetch_new_data: bool = False



def get_user_inputs():
    """Collects user inputs and returns them as variables."""

    st.markdown("## Inputs")

    # ------------------
    # Ticker Input
    # ------------------

    st.markdown("#### Tickers")
    default_ticks = ' '.join(C.SECTOR_TICKERS)
    tickers_input = st.text_area(
        "Enter tickers separated by spaces (e.g., AAPL MSFT AMZN GOOGL META TSLA JPM):",
        default_ticks
    )

    tickers = [t.strip().upper() for t in tickers_input.split(" ") if t.strip()]

    TICKER_LIMIT = 50
    if len(tickers) > TICKER_LIMIT:
        st.error(f'Sorry, for now the maximum tickers allowed is {TICKER_LIMIT}. Because I am worried about abusing the API. ')
        st.stop()

    # Raise error if there are duplicates
    if len(tickers) != len(set(tickers)):
        dups = set([ticker for ticker in tickers if tickers.count(ticker) > 1])
        st.error(f"Duplicate tickers found. Please remove: {dups}")

    # ------------------
    # Weights Input
    # ------------------
    st.markdown("#### Portfolio Weights")

    equal_weights = [1 / len(tickers) * 100]  * len(tickers)
    equal_weights = [f'{round(w, 2)}' for w in equal_weights]
    equal_weights_str = " ".join(equal_weights)
    weights_msg = "Enter the target weights for each ticker. Defaults to equal-weight. Space-separated. Percentagses. Should sum to 1, e.g, 35 25 40:"
    weights_input = st.text_area(
        weights_msg,
        equal_weights_str
    )

    weights_input = weights_input.split(" ")
    # Make sure the number of weights matches the number of tickers
    if len(weights_input) != len(tickers):
        st.error(f"Number of weights does not match number of tickers. Please provide a weight for each ticker.")
        st.stop()

    # Convert to floats
    weights_input = [float(w)/100 for w in weights_input]

    # Validat that weights are closish to 1

    DIFF_THRESHOLD = .05
    if abs(1 - sum(weights_input)) > DIFF_THRESHOLD:

        st.error(f"Your weights do not sum to 1. Please ensure they sum to 1. Current sum: {sum(weights_input)}")
        st.stop()
    
    # Rescale the weights anyway (to handle when they are super close)
    weights_input = [w / sum(weights_input) for w in weights_input]

    # ------------------
    # Date Selection
    # ------------------

    st.markdown("#### Date Range")

    # Date range selection dropdown
    date_option = st.selectbox(
        "Select a lookback range (this just assists in picking start date):",
        ["Custom", "1D", "YTD", "1 Year", "3 Years", "5 Years", "10 Years", "15 Years"],
        index=2,  # Default to "YTD"
    )

    # Automatically update start date based on selection
    date_dict = {
        "1D": DynamicDates.day_before_yesterday(),
        "YTD": DynamicDates.prior_year_end(),
        "1 Year": DynamicDates.one_year_ago(),
        "3 Years": DynamicDates.three_years_ago(),
        "5 Years": DynamicDates.five_years_ago(),
        "10 Years": DynamicDates.ten_years_ago(),
        "15 Years": DynamicDates.fifteen_years_ago(),
    }


    start_date_default = date_dict.get(date_option,None)
    if start_date_default is None:
        # If the default is none, 
        start_date_default = DynamicDates.day_before_yesterday()

    # Start date (users can override)
    start_date = st.date_input("Start Date (assumes you invest at close of this date):", start_date_default)

    # End Date Selection
    # if date_option != 'Custom':
    end_date = st.date_input("End Date (assumes you liquidate at close of this date):", DynamicDates.yesterday())
    # else:
    #     end_date = st.date_input("End Date (Assumes you liquidate at close of this date):")
    # ------------------
    # Rebalance Frequency Selection
    # ------------------

    st.markdown("#### Rebalancing Options")
    rebalance_freq = st.selectbox(
        "Select rebalance frequency:",
        [
            "YE - Year end",
            "YS - Year start",
            "QS - Quarter start",
            "QE - Quarter end",
            "MS - Month start",
            "ME - Month end",
            "W - Weekly",
            "D - Calendar day",
        ],
        index=3  # Default to "QE - Quarter end"
    )

    rebalance_freq = rebalance_freq.split(" - ")[0]  # Extract alias


    # ------------------
    # Benchmark Selection
    # ------------------
    st.markdown("#### Benchmark")
    benchmark = st.selectbox(
        "Select a benchmark (only used to calculate beta):",
        ["SPY", "IWM","QQQ","BND"],
        index=0
    )



    # ------------------
    # Portfolio Name
    # ----------------
    port_name = st.text_input("Enter a name for your portfolio:", "Port")
    # st.markdown('---')

    # Add a toggle to fetch new date or use old
    # st.markdown("#### Force Fetch New Data")
    fetch_new_data = False
    # st.write("If you need to fetch new data (not using data that is cached), toggle the switch below. You should probably never need to do this.")
    # fetch_new_data = st.toggle("Query Updated Data", value=False)

    
    
    clean_inputs = CleanInputs(
        tickers=tickers,
        weights=weights_input,
        start_date=start_date,
        end_date=end_date,
        port_name=port_name,
        rebalance_freq=rebalance_freq,
        bench_ticker=benchmark,
        fetch_new_data=fetch_new_data
    )
    return clean_inputs
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\metrics.py

```py
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

def calculate_beta(rets:pd.Series,bench_rets:pd.Series) -> float:
    '''Calculate the beta of the portfolio returns against the benchmark returns.'''
    
    # Join series together to have matching dates
    both_rets = pd.concat([rets,bench_rets],axis=1).dropna()

    cov_matrix = both_rets.cov()
    beta = cov_matrix.iloc[0,1] / cov_matrix.iloc[1,1]

    return beta

def calculate_alpha(returns:pd.Series,bench_rets:pd.Series) -> float:
    '''Calculate annualzied alpha.'''

    # Create a cleaneddataframe so we can feed it into statsmodels
    data = pd.concat([returns,bench_rets],axis=1).dropna()
    data.columns = ['port','bench']
    model = smf.ols('port ~ bench',data=data).fit()
    alpha = model.params['Intercept'] * 252

    return alpha


def upside_capture(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    '''Calculate the upside capture ratio using average returns.'''
    
    # Join series together to align dates
    both_rets = pd.concat([port_rets, bench_rets], axis=1).dropna()
    up_market_rets = both_rets[both_rets.iloc[:, 1] > 0]  # Use only periods where the benchmark is positive

    # Calculate average returns
    port_avg_ret = up_market_rets.iloc[:, 0].mean()
    bench_avg_ret = up_market_rets.iloc[:, 1].mean()

    # Avoid division by zero
    if bench_avg_ret == 0:
        return np.nan

    # Calculate upside capture ratio
    up_capture = port_avg_ret / bench_avg_ret
    return up_capture


def downside_capture(port_rets: pd.Series, bench_rets: pd.Series) -> float:
    '''Calculate the downside capture ratio using average returns.'''
    
    # Join series together to align dates
    both_rets = pd.concat([port_rets, bench_rets], axis=1).dropna()
    down_market_rets = both_rets[both_rets.iloc[:, 1] < 0]  # Use only periods where the benchmark is negative

    # Calculate average returns
    port_avg_ret = down_market_rets.iloc[:, 0].mean()
    bench_avg_ret = down_market_rets.iloc[:, 1].mean()

    # Avoid division by zero
    if bench_avg_ret == 0:
        return np.nan

    # Calculate downside capture ratio
    down_capture = port_avg_ret / bench_avg_ret
    return down_capture


def get_downside_deviation(returns, target=0):
    downside_diff = np.maximum(0, target - returns)
    squared_diff = np.square(downside_diff)
    mean_squared_diff = np.nanmean(squared_diff)
    dd = np.sqrt(mean_squared_diff) * np.sqrt(252)
    return dd


def get_max_drawdown(returns:pd.Series) -> float:

    wealth_index = (1 + returns).cumprod().array

    # Insert a wealth index of 1 at the beginning to make the calculation work
    wealth_index = np.insert(wealth_index,0,1)
    # Get the cumulative max
    cum_max = np.maximum.accumulate(wealth_index)
    max_dd = ((wealth_index / cum_max) - 1).min()

    return max_dd


def calculate_metrics(returns:pd.Series,bench_rets:pd.Series) -> dict:
    '''Calculate the key metrics for a given series of returns. Assumes returns are daily.'''

    total_ret = (1+returns).prod() - 1
    cagr = (total_ret + 1) ** (252 / returns.shape[0]) - 1
    vol = returns.std() * np.sqrt(252)
    max_dd = get_max_drawdown(returns)

    # Sharpe Ratio
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    alpha = calculate_alpha(returns,bench_rets)


    beta = calculate_beta(returns,bench_rets)
    downside_deviation = get_downside_deviation(returns)

    up_capture = upside_capture(returns,bench_rets)
    down_capture = downside_capture(returns,bench_rets)

    metrics = {
        'Total Return': total_ret,
        'CAGR': cagr,
        'Volatility': vol,
        'Sharpe': sharpe,
        'Max Drawdown': max_dd,
        'Beta': beta,
        'Alpha': alpha,
        'Downside Deviation': downside_deviation,
        'Up Capture': up_capture,
        'Down Capture': down_capture

    }

    return pd.Series(metrics)

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\requirements.txt

```txt
pandas
numpy
datetime
yfinance
streamlit
plotly
statsmodels
matplotlib

```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\results.py

```py
import pandas as pd
import streamlit as st
import plotly.express as px
import datetime as dt

import streamlit_app.data_engine as dd
import backtester as bt
import streamlit_app.inputs as inputs
import metrics
import utils


# Utility Functions
def format_as_percent(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Format specified columns as percentages."""
    for column in columns:
        df[column] = df[column].map('{:.2%}'.format)
    return df


def plot_line_chart(df: pd.DataFrame, title: str, yaxis_title: str) -> None:
    fig = px.line(df, title=title)
    fig.update_yaxes(tickformat=".2%", title_text=yaxis_title)
    fig.update_xaxes(title_text="Date")
    st.plotly_chart(fig)


def plot_bar_chart(df: pd.Series, title: str, yaxis_title: str) -> None:
    fig = px.bar(df, title=title)
    fig.update_yaxes(tickformat=".2%", title_text=yaxis_title)
    fig.update_xaxes(title_text="")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)


def display_results(backtest:bt.Backtester,data:dd.DataEngine, cleaned_inputs:inputs.CleanInputs) -> None:

    start_dt = pd.to_datetime(cleaned_inputs.start_date)

    # Security returns should be after the start date (non inclusive) and before the end date (inclusive)
    rets_filered_df = data.rets_df[data.rets_df.index > start_dt].loc[:cleaned_inputs.end_date]
    security_rets_df = rets_filered_df[cleaned_inputs.tickers]
    bench_rets = rets_filered_df[cleaned_inputs.bench_ticker]
    all_rets_df = pd.concat([backtest.port_returns, security_rets_df], axis=1)
    security_prices_df = data.price_df[data.price_df.index > start_dt].loc[:cleaned_inputs.end_date]
    security_prices_df = security_prices_df[cleaned_inputs.tickers]

    bench_rets = rets_filered_df[cleaned_inputs.bench_ticker]

    st.markdown("### Cumulative Returns")

    cum_rets_df = (1 + all_rets_df).cumprod() - 1

    plot_line_chart(cum_rets_df, "Cumulative Returns", "Cumulative Returns")
    
    # Bar plot of total return
    total_rets = cum_rets_df.iloc[-1].sort_values(ascending=False)
    plot_bar_chart(total_rets, "Total Returns", "Total Return")

    # Add on yearly returns, if we have enough data
    annual_rets = all_rets_df.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    if len(annual_rets) > 1:
        st.markdown("#### Annual Returns")        
        annual_rets.index = annual_rets.index.year
        annual_rets = annual_rets.T
        # Format these returns as a heatmap each year
        annual_rets = annual_rets.style.format("{:.2%}").background_gradient(cmap='RdYlGn', axis=1)


        st.write(annual_rets)

    # Volatility
    # Display the vol in a bar chart in the same order as the total rets
    st.markdown("### Volatility")    
    total_vol = all_rets_df.std() * 252 ** 0.5

    total_vol = total_vol[total_rets.index]
    plot_bar_chart(total_vol, "Total Period Annualized Volatility", "Volatility")

    # If you have enough data, plot the rolling vol
    ROLLING_WINDOW = 252
    if len(all_rets_df) > ROLLING_WINDOW:
    
        rolling_vols = all_rets_df.rolling(window=252).std() * 252 ** 0.5
        rolling_vols = rolling_vols.dropna()
        plot_line_chart(rolling_vols, "Rolling 1-Year Volatility", "Volatility")

    #---------------------------
    # Metrics
    #---------------------------

    st.markdown("### Performance Metrics")    
    metrics_df = all_rets_df.apply(metrics.calculate_metrics, args=(bench_rets,),axis=0)
    # We want to apply lots of fun formatting to the metrics
    metrics_pretty_df = metrics_df.T.copy()
    COLS_TO_PRETTIFY = ['Total Return', 'CAGR', 'Volatility', 'Max Drawdown', 'Alpha', 'Downside Deviation']
    metrics_pretty_df = format_as_percent(metrics_pretty_df, COLS_TO_PRETTIFY)
    # Format the following columns to onlu 2 decimal places
    DECIMAL_COLS = ['Beta', 'Sharpe', 'Up Capture', 'Down Capture']
    metrics_pretty_df[DECIMAL_COLS] = metrics_pretty_df[DECIMAL_COLS].map('{:.2f}'.format)

    st.write(metrics_pretty_df)

    # ----------------------------
    # Correlation Matrix
    # ----------------------------
    st.markdown("### Correlation Matrix")
    corr = all_rets_df.corr()
    corr_pretty_df = corr.copy()
    # # corr_pretty_df = corr_pretty_df.applymap('{:.2f}'.format)
    corr_pretty_df = corr.style.format("{:.2f}").background_gradient(cmap='coolwarm', vmin=-1, vmax=1)
    st.write(corr_pretty_df)


    # Portfolio Weights Over Time
    st.markdown("### Portfolio Weights Over Time")
    plot_line_chart(backtest.weights_df, "Portfolio Weights Over Time", "Weight")


    # # Returns
    # st.markdown("### Individual Returns")
    # # Add a tab for each of the securities and show a plot of it's indivdiual cumualtive return
    # ret_tabs = st.tabs(cum_rets_df.columns.to_list())
    # for ticker, tab in zip(cum_rets_df.columns, ret_tabs):
    #     with tab:
    #         total_ret = cum_rets_df[ticker].iloc[-1]
    #         st.write(f"Total Return: {total_ret:.2%}")
    #         plot_line_chart(cum_rets_df[ticker], f"{ticker} Cumulative Return", "Cumulative Return")

    #         # Add on another bar chart that shows the return on different time periods.


    st.markdown("### Individual Prices")
    st.write('_These prices are not adjusted for splits or dividends_')
    price_tabs = st.tabs(security_prices_df.columns.to_list())
    for ticker, tab in zip(security_prices_df.columns, price_tabs):
        with tab:
            fig = px.line(security_prices_df[ticker], title=f"{ticker} Prices")
            # Format in dollars
            fig.update_yaxes(title_text="Price", tickprefix="$")
            fig.update_xaxes(title_text="Date")
            st.plotly_chart(fig)


    

    # # ----------------------------
    # # Display Raw Data
    # # ----------------------------

    st.markdown("## Raw Data Reference")

    st.markdown("### Rebalance Dates")
    dates = backtest.rebalance_dates
    dates.name = 'Rebalance Dates'
    dates = pd.Series(dates.date, name='Rebalance Dates')
    st.write(dates)



    st.markdown("### Raw Returns")
    rets_df = utils.convert_dt_index(all_rets_df)
    # Format the returns as percentages and color code them. Positive returns are green, negative are red.
    styled_df = rets_df.style.format("{:.2%}").map(utils.color_returns)
    st.write(styled_df)

    # st.markdown("### Portfolio History")
    # st.write(utils.convert_dt_index(backtest.portfolio_history_df))

    st.markdown("### Raw Portfolio Weights")
    weights_df = utils.convert_dt_index(backtest.weights_df)
    weights_df = weights_df.map('{:.2%}'.format)
    st.write(weights_df)


if __name__ == '__main__':

    data = dd.DataEngine.load_saved_data() 
    back = bt.Backtester(data_blob=data,tickers=['AAPL','MSFT'],weights=[0.5,0.5],start_date='2010-01-01',end_date='2020-01-01')

    back.run_backtest()


    cleaned_inputs = inputs.CleanInputs(tickers=['AAPL','MSFT'],weights='0.5,0.5',start_date=dt.datetime(2010,1,1),end_date=dt.datetime(2020,1,1),port_name='Portfolio',rebalance_freq='QE',fetch_new_data=False,bench_ticker='SPY')

    display_results(back,data,cleaned_inputs)
```

## C:\Users\User\OneDrive\Desktop\Code\BG_Mono\streamlit_app\utils.py

```py
import pandas as pd
import datetime as dt

def convert_dt_index(df:pd.DataFrame) -> pd.DataFrame:
    '''Convert the index of a DataFrame from datetime to date'''
    df.index = pd.to_datetime(df.index).date
    return df

def color_returns(val):
    color = "green" if val > 0 else "red"
    return f"color: {color}"


class DynamicDates:
    @classmethod
    def today(cls):
        return dt.datetime.today()

    @classmethod
    def yesterday(cls):
        return cls.today() - dt.timedelta(days=1)

    @classmethod
    def day_before_yesterday(cls):
        return cls.today() - dt.timedelta(days=2)

    @classmethod
    def prior_year_end(cls):
        return dt.datetime(cls.today().year - 1, 12, 31)

    @classmethod
    def one_year_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 1)

    @classmethod
    def three_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 3)

    @classmethod
    def five_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 5)

    @classmethod
    def ten_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 10)

    @classmethod
    def fifteen_years_ago(cls):
        return cls.yesterday().replace(year=cls.today().year - 15)


if __name__ == '__main__':
    print("Today:", DynamicDates.today())
    print("Yesterday:", DynamicDates.yesterday())
    print("One Year Ago:", DynamicDates.one_year_ago())

```

