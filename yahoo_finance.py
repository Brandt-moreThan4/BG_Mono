import pandas as pd
import yfinance as yf
import os


def get_yf_rets(tickers: list[str]) -> pd.DataFrame:
    raw_data_df = yf.download(tickers, group_by='ticker', auto_adjust=False, actions=False)
    
    rets_df = clean_data(raw_data_df)
    return rets_df

def clean_data(raw_data_df) -> pd.DataFrame:
    df = raw_data_df.copy()
    df.index = pd.to_datetime(df.index)
    
    # Grab the actual price df
    price_df = df.loc[:, (slice(None), 'Close')]
    price_df.columns = price_df.columns.droplevel(1)

    # Grab the adjusted price df
    adjusted_prices_df = df.loc[:, (slice(None), 'Adj Close')].copy()
    adjusted_prices_df.columns = adjusted_prices_df.columns.droplevel(1)
    
    # Forward fill any nulls with adjusted prices
    adjusted_prices_df.ffill(inplace=True)

    # Calculate returns    
    rets_df = adjusted_prices_df.pct_change(fill_method=None)

    # Sort the columns alphabetically
    rets_df = rets_df[sorted(rets_df.columns)].copy()
    
    return rets_df

def test_function():
    # Test function to ensure the module works correctly
    print('I Ran!!')

if __name__ == '__main__':

    TICKERS = ['AAPL', 'MSFT', 'GOOGL']  # Example tickers
    rets_df = get_yf_rets(TICKERS)
    print('Done!')