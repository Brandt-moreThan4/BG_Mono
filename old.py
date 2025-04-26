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