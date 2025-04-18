from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import pandas as pd

import fred_fun as ff

# Paths
TEMPLATE_DIR = Path("templates")
OUTPUT_HTML = Path("output") / "fred_dashboard_1.html"
SNAPSHOT_FILE = Path("output") / "fred_dashboard_1.xlsx"


FORMAT_COLS = ['decimals', 'show_percent', 'use_commas', 'show_dollar']

def format_value(val, decimals=1, show_percent=False, use_commas=True, show_dollar=False):
    if pd.isnull(val):
        return "-"

    try:

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


# Load data
# df = ff.create_fred_snapshot(pull_new_data=True)
df = ff.create_fred_snapshot(pull_new_data=False)
# Convert the datetime to a date
df['Date'] = df['Date'].map(lambda x: x.date())



fred_map_df = pd.read_excel(ff.MASTER_FILE, sheet_name='master').set_index('display_name')

# df = df.merge(fred_map_df, left_index=True, right_on='display_name')

# Prepare rows for Jinja template
rows = []
for index, row in df.iterrows():
    meta_data = fred_map_df.loc[index]
    format_meta = meta_data[FORMAT_COLS].to_dict()
    rows.append({
        "display_name": index,
        "latest_value": format_value(row["Value"]["latest"], **format_meta),
        "lag_1_value": format_value(row["Value"]["lag_1"], **format_meta),
        "lag_2_value": format_value(row["Value"]["lag_2"], **format_meta),
        "latest_date": row["Date"]["latest"],
        "lag_1_date": row["Date"]["lag_1"],
        "lag_2_date": row["Date"]["lag_2"],
        'url': meta_data['link'],
    })


# Jinja env
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
template = env.get_template("full_report.html")

# Render
html = template.render(rows=rows)

# Save output
OUTPUT_HTML.parent.mkdir(exist_ok=True)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"✅ Report generated at: {OUTPUT_HTML}")