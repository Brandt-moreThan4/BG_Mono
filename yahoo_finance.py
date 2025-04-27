import pandas as pd
import yfinance as yf


class YFinanceData:
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self._raw_data_df = None
        self._price_data_df = None
        self._adjusted_price_data_df = None
        self._returns_df = None

        self.run()

    def fetch_data(self) -> None:
        self._raw_data_df = yf.download(self.tickers, group_by='ticker', auto_adjust=False, actions=False)
        self._raw_data_df.index = pd.to_datetime(self._raw_data_df.index)

    def clean_data(self) -> None:
        if self._raw_data_df is None:
            raise ValueError("No raw data to clean. Please fetch data first.")

        self._price_data_df = self._raw_data_df.loc[:, (slice(None), 'Close')]
        self._price_data_df.columns = self._price_data_df.columns.droplevel(1)

        self._adjusted_price_data_df = self._raw_data_df.loc[:, (slice(None), 'Adj Close')].copy()
        self._adjusted_price_data_df.columns = self._adjusted_price_data_df.columns.droplevel(1)
        self._adjusted_price_data_df.ffill(inplace=True)

        self._returns_df = self._adjusted_price_data_df.pct_change(fill_method=None)
        self._returns_df = self._returns_df[sorted(self._returns_df.columns)].copy()

    def run(self) -> None:
        self.fetch_data()
        self.clean_data()

    @property
    def raw_data(self) -> pd.DataFrame:
        if self._raw_data_df is None:
            raise ValueError("Raw data not available.")
        return self._raw_data_df.copy()

    @property
    def price_data(self) -> pd.DataFrame:
        if self._price_data_df is None:
            raise ValueError("Price data not available.")
        return self._price_data_df.copy()

    @property
    def adjusted_price_data(self) -> pd.DataFrame:
        if self._adjusted_price_data_df is None:
            raise ValueError("Adjusted price data not available.")
        return self._adjusted_price_data_df.copy()

    @property
    def returns(self) -> pd.DataFrame:
        if self._returns_df is None:
            raise ValueError("Returns not available.")
        return self._returns_df.copy()

    def __repr__(self) -> str:
        return f"YFinanceData(tickers={self.tickers})"
    
    def __str__(self) -> str:
        return f"YFinanceData with {len(self.tickers)} tickers: {', '.join(self.tickers)}"


if __name__ == '__main__':
    TICKERS = ['AAPL', 'MSFT', 'GOOGL']
    yf_data = YFinanceData(TICKERS)
    returns_df = yf_data.returns
    print('Done!')
