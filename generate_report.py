from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

import utils
import fred_fun as ff

# Paths
TEMPLATE_DIR = Path("templates")
OUTPUT_HTML = Path("output") / "fred_dashboard_1.html"
SNAPSHOT_FILE = Path("output") / "fred_dashboard_1.xlsx"
OUTPUT_FOLDER = Path("output")
IMAGES_FOLDER = OUTPUT_FOLDER / "images"

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



def generate_inflation_chart(inflation_df:pd.DataFrame) -> None:

    # Generate the inflation chart
    fig, ax = plt.subplots()
    df_12_month_plot = inflation_df.copy()
    name_mapper = master_fred_map_df.set_index('fred_id')['display_name']
    df_12_month_plot.columns = df_12_month_plot.columns.map(name_mapper)
    df_12_month_plot.plot(ax=ax)

    # Se the title and make it big
    ax.set_title("12-Month Inflation", fontsize=20, fontweight='bold')

    # Add the legend
    ax.legend(loc='upper left')
    
    # Format the y-axis to show percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Remove the xaxis label
    ax.set_xlabel("")


    # Save the chart
    chart_path = IMAGES_FOLDER / "inflation_chart.png"
    fig.savefig(chart_path)
    # fig.savefig(chart_path)    
    # plt.close(fig)




def generate_inflation_report() -> str:


    # Pull the inflation data from FRED
    inflation_data = ff.get_fred_data(fred_ids=['CPIAUCSL', 'CPILFESL', 'PCEPI', 'PCEPILFE'])

    # Convert the dataframe to wide format to make it easier to work with
    df = inflation_data.pivot(index='date', columns='fred_id', values='value')


    # Compute % Changes
    df_12_month = df.pct_change(periods=12)
    df_1_month = df.pct_change(periods=1)

    generate_inflation_chart(df_12_month)

    # Render the Jinja template with the inflation data
    inflation_template = jinja_env.get_template("3_inflation.html")
    html = inflation_template.render()

    return html 



def generate_report() -> None:


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