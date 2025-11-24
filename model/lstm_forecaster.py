# model/lstm_forecaster.py

import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

def train_lstm_model(data: pd.DataFrame, steps: int = 15):
    """
    Trains an LSTM model on the stock's 'Close' price and forecasts for N future steps.
    """
    if 'Close' not in data.columns:
        raise ValueError("Data must contain a 'Close' column.")

    # Prepare close price series
    dataset = data[['Close']].values
    training_data_len = int(len(dataset) * 0.95)

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)

    # Training data
    training_data = scaled_data[:training_data_len]
    X_train, y_train = [], []

    for i in range(60, len(training_data)):
        X_train.append(training_data[i - 60:i, 0])
        y_train.append(training_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the LSTM model
    model = keras.models.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mae", metrics=[keras.metrics.RootMeanSquaredError()])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Prepare input for forecasting
    last_60 = scaled_data[-60:]
    input_seq = np.reshape(last_60, (1, last_60.shape[0], 1))

    preds_scaled = []
    for _ in range(steps):
        pred = model.predict(input_seq, verbose=0)[0][0]
        preds_scaled.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds
