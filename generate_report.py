from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import pandas as pd

import fred_fun as ff

# Paths
TEMPLATE_DIR = Path("templates")
OUTPUT_HTML = Path("output") / "fred_dashboard_1.html"
SNAPSHOT_FILE = Path("output") / "fred_dashboard_1.xlsx"

# Load data
df = ff.create_fred_snapshot().reset_index()


# Prepare rows for Jinja template
rows = []
for _, row in df.iterrows():
    rows.append({
        "display_name": row["display_name"].value,
        "latest_value": row["Value"]["latest"],
        "lag_1_value": row["Value"]["lag_1"],
        "lag_2_value": row["Value"]["lag_2"],
        "latest_date": row["Date"]["latest"],
        "lag_1_date": row["Date"]["lag_1"],
        "lag_2_date": row["Date"]["lag_2"],
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