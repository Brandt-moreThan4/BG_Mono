from fredapi import Fred
from pathlib import Path
import pandas as pd
import xlwings as xw
import logging
import time

fred = Fred(api_key='37eb22bada238c97f282715480e7d897')


MASTER_FILE = Path('reference') / 'fred.xlsx'
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




def create_fred_snapshot() -> pd.DataFrame:

    fred_map_df = pd.read_excel(MASTER_FILE, sheet_name='master')
    
    # Grab the list of data points to include in this snapshot
    data_to_pull = fred_map_df[fred_map_df['dashboard_1'] == True]['fred_id']

    master_results = {}
    for data_series in data_to_pull:
        logging.info(f'Pulling data for: "{data_series}" ')
        try:
            # Grab the data and make sure it is sorted by date
            data = fred.get_series(data_series).sort_index(ascending=True)

            data_summary = {
                'latest': data.iloc[-1],
                'lag_1': data.iloc[-2],
                'lag_2': data.iloc[-3],
                'latest_date': data.index[-1].date(),
                'lag_1_date': data.index[-2].date(),
                'lag_2_date': data.index[-3].date()
            }
            master_results[data_series] = data_summary

            # Sleep a sliver of time to avoid hitting the API too hard
            time.sleep(SLEEP_TIME)

        except Exception as e:
            # Don't want to stop the entire process if one data series fails to pull
            # Log the error and continue with the next data series
            logging.error(f'Error pulling data for {data_series}. Continuing. Exception message: {e}. ')

    # Create a pretty DataFrame from the results
    results_df = pd.DataFrame(master_results).T
    results_df.columns = pd.MultiIndex.from_product([['Value', 'Date'],['latest', 'lag_1', 'lag_2']])

    # Add on the pretty name columns from master file
    results_df.index = results_df.index.map(fred_map_df.set_index('fred_id')['display_name'])  
    results_df.index.name = 'display_name'

    results_df.to_excel(DASHBOARD_1_EXPORT_PATH, sheet_name='dashboard_1', index=True, header=True)
    logging.info(f'Exported data to {DASHBOARD_1_EXPORT_PATH}')

    return results_df


if __name__ == "__main__":
    
    
    create_fred_snapshot()
    
    wb = xw.Book(DASHBOARD_1_EXPORT_PATH)
    print('Done!!!')