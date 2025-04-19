import pandas as pd
from pathlib import Path
from matplotlib import cycler
import matplotlib as mpl
from matplotlib import pyplot as plt

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