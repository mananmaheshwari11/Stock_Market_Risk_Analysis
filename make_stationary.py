import pandas as pd
from statsmodels.tsa.stattools import adfuller

def make_stationary(df, col='Close'):
    df = df.copy()

    if 'Date' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.columns[0]: 'Date'})
        else:
            df['Date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(x) for x in c if x]).strip() for c in df.columns]
    col_candidates = [c for c in df.columns if col.lower() in c.lower()]
    if not col_candidates:
        print("Available columns:", list(df.columns))
        raise KeyError(f"Column '{col}' not found in dataframe!")
    col = col_candidates[0]
    df['Returns'] = df[col].astype(float).pct_change()
    df.dropna(subset=['Returns'], inplace=True)

    try:
        adf_result = adfuller(df['Returns'])
        print(f"ADF Statistic: {adf_result[0]:.4f}")
        print(f"p-value: {adf_result[1]:.4f}")
        if adf_result[1] < 0.05:
            print("Stationary — Ready for forecasting!")
        else:
            print("Not stationary — consider differencing.")
    except Exception as e:
        print(f"ADF test failed: {e}")

    print(f"Stationarity processing done for {col}")
    print(f"Final data shape: {df.shape}")

    return df[['Date', col, 'Returns']]