# data_fetcher.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from config.stocks_config import TOP_50_STOCKS
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Local in-memory cache (keyed by symbol,start,end,interval)
_CACHE = {}

# ------------------------------
# basic helpers / config lookup
# ------------------------------
def get_stock_config(symbol: str):
    """
    Return config info for symbol from TOP_50_STOCKS.
    If not found, raise ValueError.
    """
    info = TOP_50_STOCKS.get(symbol)
    if not info:
        raise ValueError(f"Symbol '{symbol}' not configured in TOP_50_STOCKS. Add it to config/stocks_config.py.")
    return info

def _cache_key(symbol, start, end, interval):
    return f"{symbol}__{start or ''}__{end or ''}__{interval}"

# ------------------------------
# download function (full OHLCV)
# ------------------------------
def download_price_series(symbol: str, start: str = None, end: str = None, interval: str = "1d"):
    """
    Download full OHLCV series using yfinance.
    Returns DataFrame with columns: ds (date), Open, High, Low, Close, Adj Close, Volume.
    Caches results in _CACHE to avoid repeated downloads during a session.
    """
    key = _cache_key(symbol, start, end, interval)
    if key in _CACHE:
        return _CACHE[key].copy()

    if end is None:
        end = datetime.utcnow().date().isoformat()
    if start is None:
        start = "2015-01-01"

    try:
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data for {symbol}: {e}")

    if df is None or df.empty:
        raise RuntimeError(f"No data returned from yfinance for {symbol}. Check symbol and date range.")

    # Reset index to make Date a column
    df = df.reset_index()
    
    # Handle multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten multi-index by taking first level (main column names)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Standardize column names
    column_mapping = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in ['date', 'datetime']:
            column_mapping[col] = 'ds'
        elif col_lower == 'open':
            column_mapping[col] = 'Open'
        elif col_lower == 'high':
            column_mapping[col] = 'High'
        elif col_lower == 'low':
            column_mapping[col] = 'Low'
        elif col_lower == 'close':
            column_mapping[col] = 'Close'
        elif col_lower in ['adj close', 'adj_close', 'adjclose']:
            column_mapping[col] = 'Adj Close'
        elif col_lower == 'volume':
            column_mapping[col] = 'Volume'
    
    df = df.rename(columns=column_mapping)
    
    # Ensure we have required columns
    if 'ds' not in df.columns:
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'ds'})
        else:
            raise RuntimeError(f"Cannot find date column for {symbol}")
    
    # Ensure we have Close price
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        else:
            raise RuntimeError(f"Missing 'Close' or 'Adj Close' columns for {symbol}. Available columns: {df.columns.tolist()}")
    
    # Ensure we have Adj Close
    if 'Adj Close' not in df.columns:
        if 'Close' in df.columns:
            df['Adj Close'] = df['Close']
    
    # Fill in missing OHLC if not present
    if 'Open' not in df.columns:
        df['Open'] = df['Close']
    if 'High' not in df.columns:
        df['High'] = df['Close']
    if 'Low' not in df.columns:
        df['Low'] = df['Close']
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    
    # Select and order columns
    expected_cols = ['ds', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df = df[expected_cols].copy()

    # Convert ds to datetime and sort
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df = df.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)
    
    # Remove any rows with missing Close prices
    df = df.dropna(subset=['Close']).reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"No valid data remaining for {symbol} after cleaning")

    _CACHE[key] = df.copy()
    return df

# ------------------------------
# Prepare Prophet dataframe
# ------------------------------
def prepare_prophet_dataframe(symbol: str, start: str = None, end: str = None, include_regressors: bool = True):
    """
    Prepare a DataFrame suitable for Prophet training.
    Returns df_prophet with columns: ds, y, and optionally market_index, ma_50, ma_200
    """
    cfg = get_stock_config(symbol)
    region = cfg.get('region', 'US')
    currency = cfg.get('currency', 'USD')
    index_symbol = cfg.get('index')

    # Download full OHLCV for stock
    stock_df = download_price_series(symbol, start=start, end=end)
    if stock_df is None or stock_df.empty:
        raise RuntimeError(f"No valid data found for {symbol}")

    if 'Close' not in stock_df.columns:
        raise RuntimeError(f"'Close' column missing for {symbol} data.")

    # Create base dataframe with ds and y
    df_prophet = stock_df[['ds', 'Close']].copy()
    df_prophet = df_prophet.rename(columns={'Close': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    df_prophet = df_prophet.sort_values('ds').reset_index(drop=True)
    
    # Remove any NaN values in target
    df_prophet = df_prophet.dropna(subset=['y']).reset_index(drop=True)

    if df_prophet.empty:
        raise RuntimeError(f"No valid data after cleaning for {symbol}")

    # Add regressors if requested
    if include_regressors:
        # Add moving averages
        df_prophet['ma_50'] = df_prophet['y'].rolling(window=50, min_periods=1).mean()
        df_prophet['ma_200'] = df_prophet['y'].rolling(window=200, min_periods=1).mean()
        
        # Add market index if available
        if index_symbol:
            try:
                index_df = download_price_series(index_symbol, start=start, end=end)
                if index_df is not None and not index_df.empty:
                    # Prepare index series
                    index_series = index_df[['ds', 'Close']].copy()
                    index_series = index_series.rename(columns={'Close': 'market_index'})
                    index_series['ds'] = pd.to_datetime(index_series['ds'])
                    
                    # Merge with main dataframe - use left join to preserve all stock data
                    df_prophet = pd.merge(df_prophet, index_series, on='ds', how='left')
                    
                    # Forward fill then backward fill any missing index values
                    df_prophet['market_index'] = df_prophet['market_index'].ffill().bfill()
                    
                    # If still have NaNs, use the stock price as fallback
                    if df_prophet['market_index'].isna().any():
                        df_prophet['market_index'] = df_prophet['market_index'].fillna(df_prophet['y'])
                else:
                    # Index fetch returned no data
                    print(f"[WARN] No index data available for {index_symbol}, using stock price as fallback")
                    df_prophet['market_index'] = df_prophet['y']
            except Exception as e:
                # Index fetch failed
                print(f"[WARN] Failed to fetch index data for {index_symbol}: {e}")
                df_prophet['market_index'] = df_prophet['y']
        else:
            # No index symbol configured
            df_prophet['market_index'] = df_prophet['y']
        
        # Ensure no NaN values in regressors
        df_prophet['ma_50'] = df_prophet['ma_50'].fillna(df_prophet['y'])
        df_prophet['ma_200'] = df_prophet['ma_200'].fillna(df_prophet['y'])
        
        # Final column selection
        df_prophet = df_prophet[['ds', 'y', 'market_index', 'ma_50', 'ma_200']].copy()
    else:
        # No regressors - just ds and y
        df_prophet = df_prophet[['ds', 'y']].copy()
    
    # Final check for any remaining NaNs in target
    df_prophet = df_prophet.dropna(subset=['y']).reset_index(drop=True)

    metadata = {
        'symbol': symbol,
        'region': region,
        'currency': currency,
        'index_symbol': index_symbol,
        'first_date': df_prophet['ds'].min().isoformat() if not df_prophet.empty else None,
        'last_date': df_prophet['ds'].max().isoformat() if not df_prophet.empty else None,
        'total_rows': len(df_prophet)
    }
    return df_prophet, metadata


# ------------------------------
# Prepare Transformer dataframe
# ------------------------------
def prepare_transformer_dataframe(symbol: str, seq_length: int = 60, start: str = None, end: str = None):
    """
    Prepare dataframe for Transformer model training and next-day forecasting.
    
    Returns:
      - X: all sequences up to last available window
      - y: target Close price for each sequence
      - scaler: fitted MinMaxScaler
      - df: cleaned full DataFrame with all indicators
    """
    # Fetch data
    stock_df = download_price_series(symbol, start=start, end=end)
    stock_df['ds'] = pd.to_datetime(stock_df['ds'])
    stock_df = stock_df.sort_values('ds').reset_index(drop=True)

    # Check required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in stock_df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required columns {missing_cols} in data for {symbol}.")

    # Feature engineering
    df = stock_df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['MA_7'] = df['Close'].rolling(window=7, min_periods=1).mean()
    df['MA_21'] = df['Close'].rolling(window=21, min_periods=1).mean()
    df['Volatility'] = df['Returns'].rolling(window=21, min_periods=1).std()
    df['Volume_MA'] = df['Volume'].rolling(window=7, min_periods=1).mean()
    
    # Fill any remaining NaN values
    df['Returns'] = df['Returns'].fillna(0)
    df['Volatility'] = df['Volatility'].fillna(0)
    
    # Drop any rows with NaN in critical columns
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume']).reset_index(drop=True)

    if len(df) < seq_length + 1:
        raise RuntimeError(f"Not enough data for {symbol}. Need at least {seq_length + 1} rows, got {len(df)}")

    # Feature set
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'MA_7', 'MA_21', 'Volatility', 'Volume_MA']
    data_features = df[feature_cols].values.astype(float)

    # Normalize
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_features)

    # Create sequences
    def create_sequences(data, seq_len=60):
        X, y = [], []
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len:i])
            y.append(data[i, 3])  # predicting Close (index 3)
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, seq_length)
    if len(X) == 0:
        raise RuntimeError(f"Not enough data for sequence length {seq_length}. Need at least {seq_length + 1} data points.")

    return X, y, scaler, df