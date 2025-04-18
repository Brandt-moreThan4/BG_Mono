from fredapi import Fred

fred = Fred(api_key='37eb22bada238c97f282715480e7d897')



# def get_fred_data


if __name__ == "__main__":
    # Example usage
    data = fred.get_series('GDP')
    print(data.tail())