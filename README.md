# 📈 Stock Market Risk Analysis App

This project is an intelligent **Stock Market Risk Analysis** and **Prediction System** that leverages both **Prophet** (for time-series forecasting) and **Transformer-based models** (for deep learning-based trend analysis).  
It predicts stock prices over user-selected time periods (7 days, 3 months, or 1 year) and provides analytical insights into risk levels, helping investors and analysts make informed decisions.

---

## 🚀 Features

- 🔮 **Prophet Model** for interpretable trend, seasonality, and future price prediction.  
- 🤖 **Transformer Model** for sequence-based deep learning forecasting.  
- 📅 **Dynamic time period selection:** 7 days, 3 months, or 1 year.  
- 🌐 **Real-time data fetching** from Yahoo Finance (2015 → current date).  
- 🧠 **Fine-tuning & caching** for faster and more accurate predictions.  
- 📊 **Automatic scaling** and resampling of data for both models.  
- ⚙️ **API-ready** endpoints for integration with dashboards or UIs.  

---

## 🧩 Project Structure

stock_predictor/
│
├── stock_venv/ # Virtual environment (ignored in .gitignore)
│
├── app.py # Main Flask app containing API routes
├── data_fetcher.py # Handles Yahoo Finance data fetching and preprocessing
├── requirements.txt # Python dependencies
├── .env.example # Example environment file
├── .gitignore # Ignored files (env, pycache, models, etc.)
└── README.md # Project documentation


---

## 🔁 Data Flow


1. **User selects a stock symbol** (e.g., `AAPL`, `GOOG`) and a **time period** (7 days, 3 months, or 1 year).  
2. **Data Fetcher** pulls historical data (from `2015-01-01` → current date).  
3. **Preprocessing** prepares the data for both Prophet and Transformer models:
   - Prophet expects `ds` (date) and `y` (closing price).
   - Transformer expects normalized numeric time-series input.  
4. Models predict the future price trend.
5. **App returns** predictions with timestamps and confidence intervals.

---

## 🧠 Model Fine-tuning Details

### 🪄 Prophet Model
- Uses **additive regression** with trend, seasonality, and holiday effects.  
- Fine-tuned for:
  - `changepoint_prior_scale`: controls trend flexibility.
  - `seasonality_mode`: set to “additive” for smoother stock trends.
  - `daily`, `weekly`, and `yearly` seasonality adjustments.
- Automatically adapts to 7-day, 3-month, or 1-year prediction horizons.

### ⚡ Transformer Model
- Based on **sequence-to-sequence architecture** with attention mechanism.
- Fine-tuned parameters:
  - Sequence length (e.g., 30–60 previous days).
  - Learning rate scheduling for time-series patterns.
  - Batch normalization for stable gradients.
  - Dropout for overfitting control.
- Uses a **custom positional encoding** to retain temporal dependencies.

---

## 🧰 API Endpoints

### 🔹 Prophet Forecast
**POST** `/predict/prophet`

**Example Request (Postman):**
```json
{
  "symbol": "AAPL",
  "period": "3m"
}
{
  "symbol": "GOOG",
  "period": "1y"
}

```

**Example Response (Postman):**
```json
{
  "model": "prophet",
  "symbol": "AAPL",
  "predictions": [
    {"ds": "2025-02-04", "yhat": 215.32, "yhat_lower": 210.21, "yhat_upper": 219.80},
    {"ds": "2025-02-05", "yhat": 216.90, "yhat_lower": 212.00, "yhat_upper": 220.50}
  ],
  "status": "success"
}

{
  "model": "transformer",
  "symbol": "GOOG",
  "predictions": [
    {"date": "2026-11-04", "predicted_close": 2911.45},
    {"date": "2026-11-05", "predicted_close": 2922.10}
  ],
  "status": "success"
}
```

# Clone project
git clone https://github.com/mananmaheshwari11/Stock_Market_Risk_Analysis.git

# Setup environment
python -m venv stock_venv
stock_venv\Scripts\activate
pip install -r requirements.txt

# Run Flask app
python app.py



