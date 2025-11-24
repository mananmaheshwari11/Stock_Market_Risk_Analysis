import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend suitable for Flask

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress FutureWarnings

from flask import Flask, render_template, request, jsonify
from data_fetcher import fetch_from_yahoo, fetch_from_google
from make_stationary import make_stationary
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import date
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data', methods=['POST'])
def get_data():
    source = request.form['source']
    symbol = request.form['symbol']
    duration = request.form['duration']
    model_type = request.form.get('model', 'ARIMA')  # Default to ARIMA if not specified

    # Step 1: Fetch data
    if source == 'yahoo':
        data = fetch_from_yahoo(symbol)
    elif source == 'google':
        data = fetch_from_google(symbol)
    else:
        return jsonify({"error": "Invalid source"})

    # Step 2: Preprocess for ARIMA
    print(f"\nMaking {symbol} data stationary")
    data = make_stationary(data, 'Close')
    data.to_csv(f"data/{symbol}_stationary.csv", index=False)

    # Step 3: Set forecast duration
    if duration == '15d':
        steps = 15
    elif duration == '3m':
        steps = 90
    else:
        steps = 365

    # Step 4: Train and forecast using selected model
    if model_type.upper() == 'ARIMA':
        from model.forecaster import train_model
        model_fit = train_model(symbol)
        forecast = model_fit.forecast(steps=steps)

    elif model_type.upper() == 'LSTM':
        from model.lstm_forecaster import train_lstm_model
        forecast = train_lstm_model(data, steps=steps)
        forecast = pd.Series(forecast)

    else:
        return jsonify({"error": "Invalid model type"})

    # Step 5: Prepare dates
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.set_index('Date')

    data.index = pd.to_datetime(data.index)
    start_date = pd.to_datetime('2023-01-01')
    data = data[data.index >= start_date]

    if data.empty:
        return jsonify({"error": "No data available after filtering. Check make_stationary output."})

    forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=steps, freq='B')
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecast': forecast.values
    })

    # Step 6: Plot results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Actual Prices', color='purple')
    ax.plot(forecast_df['Date'], forecast_df['Forecast'], label=f'{model_type.upper()} Forecast', color='red', linestyle='dotted')

    plt.legend()
    plt.title(f"{symbol} {steps}-Day {model_type.upper()} Forecast (2023–Present)")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Ensure static directory exists
    os.makedirs('static', exist_ok=True)
    plot_filename = f"static/{symbol}_{model_type.lower()}_{steps}_forecast.png"
    plt.savefig(plot_filename)
    plt.close()

    # Step 7: Return chart data + plot URL
    return jsonify({
        "chart_data": {
            "actual_dates": data.index.strftime('%Y-%m-%d').tolist(),
            "actual_values": data['Close'].tolist(),
            "forecast_dates": forecast_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            "forecast_values": forecast_df['Forecast'].tolist(),
            "plot_url": plot_filename
        }
    })


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
