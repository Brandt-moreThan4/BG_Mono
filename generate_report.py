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