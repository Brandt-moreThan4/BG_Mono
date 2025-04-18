from fredapi import Fred
from pathlib import Path
import pandas as pd
import xlwings as xw
import logging
import time

fred = Fred(api_key='37eb22bada238c97f282715480e7d897')


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


def create_fred_snapshot(pull_new_data=True) -> pd.DataFrame:

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