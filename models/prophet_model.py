# models/prophet_model.py
from prophet import Prophet
import pandas as pd

def train_prophet(df: pd.DataFrame, region: str = "US", changepoint_prior_scale: float = 0.05,
                  seasonality_mode: str = 'additive', regressors: list = None, holidays: bool = True):
    """
    Train Prophet using the dataframe already prepared by data_fetcher.prepare_prophet_dataframe.
    df must contain columns: 'ds', 'y' and optional regressors e.g. 'market_index', 'ma_50'.
    regressors: list of column names in df to add as regressors.
    region: used to add_country_holidays (if supported by Prophet).
    """
    if regressors is None:
        regressors = ['market_index','ma_50','ma_200'] if {'market_index','ma_50','ma_200'}.issubset(df.columns) else []

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_mode=seasonality_mode
    )

    # add regressors present in df
    for r in regressors:
        if r in df.columns:
            m.add_regressor(r)

    # add country holidays if requested (Prophet supports several country calendars)
    if holidays:
        try:
            if region == 'US':
                m.add_country_holidays(country_name='US')
            elif region == 'IN':
                m.add_country_holidays(country_name='IN')
            # add other regions if you extend TOP_50_STOCKS
        except Exception:
            # Some Prophet builds may not have all country calendars - ignore if it fails
            pass

    # Fit
    # Prophet expects df with ds and y; regressors must be present
    m.fit(df)
    return m

def predict_prophet(model: Prophet, history_df: pd.DataFrame, periods: int, freq: str = 'D'):
    """
    model: fitted Prophet model
    history_df: original df used for training (needed to build regressors into future)
    periods: number of periods to forecast
    freq: frequency of predictions ('D' for daily)
    Returns: forecast DataFrame (ds, yhat, yhat_lower, yhat_upper)
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)

    # Propagate regressors into future: conservative approach is to carry last known value forward.
    # If you have forecasts for the regressors (e.g., index forecasts), replace this logic.
    for col in ['market_index', 'ma_50', 'ma_200']:
        if col in history_df.columns:
            last_val = history_df[col].iloc[-1]
            future[col] = last_val
    
    forecast = model.predict(future)
    
    # return only future rows (not historical)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(periods).reset_index(drop=True)