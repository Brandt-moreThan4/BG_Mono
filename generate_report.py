from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import pandas as pd
import utils

import fred_fun as ff

# Paths
TEMPLATE_DIR = Path("templates")
OUTPUT_HTML = Path("output") / "fred_dashboard_1.html"
SNAPSHOT_FILE = Path("output") / "fred_dashboard_1.xlsx"


FORMAT_COLS = ['decimals', 'show_percent', 'use_commas', 'show_dollar']

fred_map_df = pd.read_excel(ff.MASTER_FILE, sheet_name='master').set_index('display_name')


def get_macro_dashboard_data() -> list[dict]:

    # Grab the cleaned snapshot data    
    df = ff.create_fred_snapshot(pull_new_data=False)

    # Convert the datetime to a date
    df['Date'] = df['Date'].map(lambda x: x.date())

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


def get_inflation_data() -> pd.DataFrame:

    # CPIAUCSL
    # CPILFESL
    # PCEPI
    # PCEPILFES

    # Pull the inflation data from FRED
    inflation_data = ff.get_fred_data(fred_ids=['CPIAUCSL', 'CPILFESL', 'PCEPI', 'PCEPILFES'])

    # Convert the dataframe to wide format to make it easier to work with
    df = inflation_data.pivot(index='date', columns='fred_id', values='value')
    



def generate_report() -> None:

    # Jinja env
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    full_report_template = env.get_template("0_full_report.html")
    dashboard_template = env.get_template("2_macro_dash.html")
    inflation_template = env.get_template("3_inflation.html")
    gdp_template = env.get_template("4_gdp.html")


    # Render individual sections to HTML snippets
    dashboard_html = dashboard_template.render(rows=get_macro_dashboard_data())
    gdp_html = gdp_template.render()
    inflation_html = inflation_template.render()

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