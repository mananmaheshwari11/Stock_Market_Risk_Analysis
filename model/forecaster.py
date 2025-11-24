import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA
import itertools
import statsmodels.api as sm

def train_model(symbol='AAPL'):
    df = pd.read_csv(f"data/{symbol}_stationary.csv")

    if 'Date' not in df.columns:
        df.reset_index(inplace=True)  

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Date', 'Close'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.asfreq('B')  

    print("Data loaded:", df.head())

    model=sm.tsa.statespace.SARIMAX(df['Close'], order=(1,1,1),seasonal_order=(1,1,1,30))
    model_fit=model.fit()

    with open("model/arima_model.pkl", "wb") as f:
        pickle.dump(model_fit, f)

    return model_fit
