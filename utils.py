import pandas as pd
from pathlib import Path

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

