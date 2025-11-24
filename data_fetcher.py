import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from datetime import date, timedelta

def fetch_from_google(symbol):
    """
    Fetch stock data from your Google Sheet (simulating Google Finance),
    fallback to Yahoo if unavailable.
    """
    google_sheets = {
        "AAPL": "https://docs.google.com/spreadsheets/d/1BxM3XM-5wWypexUG9bEtlmhyHOoYu0HAcv0vZxxiWzc/gviz/tq?tqx=out:csv&sheet=Sheet1",
        "TSLA": "https://docs.google.com/spreadsheets/d/1BxM3XM-5wWypexUG9bEtlmhyHOoYu0HAcv0vZxxiWzc/gviz/tq?tqx=out:csv&sheet=Sheet2",
        "MSFT": "https://docs.google.com/spreadsheets/d/1BxM3XM-5wWypexUG9bEtlmhyHOoYu0HAcv0vZxxiWzc/gviz/tq?tqx=out:csv&sheet=Sheet3"
    }

    try:
        if symbol not in google_sheets:
            raise ValueError("No Google Sheet URL for this symbol")

        print(f"Attempting Google Finance fetch for {symbol}...")
        response = requests.get(google_sheets[symbol])
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        df.columns = [c.strip().capitalize().replace(" ", "_") for c in df.columns]
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')

        rename_map = {c: 'Close' for c in df.columns if 'close' in c.lower()}
        df.rename(columns=rename_map, inplace=True)

        print(f"\nGoogle data fetched successfully for {symbol}")
        df.to_csv(f"data/{symbol}_google.csv", index=False)
        return df

    except Exception as e:
        print(f"\nGoogle Finance failed for {symbol}: {e}")
        print("Falling back to Yahoo Finance instead...")
        return fetch_from_yahoo(symbol)


def fetch_from_yahoo(symbol):
    end = date.today()
    start = end - timedelta(days=365 * 5)

    print(f"Attempting Yahoo Finance fetch for {symbol}...")

    df = yf.download(symbol, start=start, end=end)
    if df.empty:
        raise ValueError("Yahoo returned empty dataset")

    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] != '' else c[1] for c in df.columns]  
    else:
        df.columns = [str(c).strip().replace('.', '_').replace(' ', '_').capitalize() for c in df.columns]


    if 'Date' not in df.columns:
        date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or 'index' in c.lower()]
        if date_col:
            df.rename(columns={date_col[0]: 'Date'}, inplace=True)
        else:
            raise KeyError("No valid Date column found after resetting index!")

    df.columns = [str(c).strip().replace('.', '_').replace(' ', '_').capitalize() for c in df.columns]

    rename_map = {c: 'Close' for c in df.columns if 'close' in c.lower()}
    df.rename(columns=rename_map, inplace=True)

    us_cal = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    my_range = pd.date_range(start=df['Date'].min(), end=end, freq=us_cal)
    missing_days = my_range.difference(df['Date'])

    print(f"\nYahoo data fetched for {symbol}")
    print(f"Missing {len(missing_days)} trading days auto-detected")

    df.to_csv(f"data/{symbol}_yahoo.csv", index=False)
    return df