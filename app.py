# app.py
import traceback
from datetime import datetime
from flask import Flask, request, jsonify

# Use the prepare functions from your services/data_fetcher.py
from data_fetcher import prepare_prophet_dataframe, prepare_transformer_dataframe

# Import model train/predict functions (names match previous assistant code)
from models.prophet_model import train_prophet, predict_prophet
from models.transformer_model import train_transformer_model, predict_future

app = Flask(__name__)

# period mapping (user input -> number of days)
PERIOD_MAPPING = {
    "7d": 7,
    "15d": 15,
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365
}

# Transformer safe cap
MAX_TRANSFORMER_STEPS = 90

# In-memory model cache
MODEL_CACHE = {
    "prophet": {},        # symbol -> {"model": ProphetModel, "trained_at": ts, "meta": metadata}
    "transformer": {}     # symbol -> {"model": tf_model, "scaler": scaler, "seq_length": int, "trained_at": ts, "meta": metadata}
}


@app.route('/')
def index():
    return jsonify({
        "message": "Stock Forecast API",
        "notes": "Use /predict/prophet and /predict/transformer. Models are cached per symbol; use 'refresh':true to force retrain."
    })


# ---------------------------
# Prophet route
# ---------------------------
@app.route('/predict/prophet', methods=['POST'])
def predict_prophet_route():
    try:
        payload = request.get_json(force=True)
        symbol = payload.get("symbol", "AAPL")
        period_key = payload.get("period", "7d")
        refresh = bool(payload.get("refresh", False))

        if period_key not in PERIOD_MAPPING:
            return jsonify({"error": "Invalid period. Choose one of: " + ", ".join(PERIOD_MAPPING.keys())}), 400
        periods = PERIOD_MAPPING[period_key]

        # Check cache
        cache_hit = (not refresh) and (symbol in MODEL_CACHE["prophet"])
        if cache_hit:
            model_info = MODEL_CACHE["prophet"][symbol]
            model = model_info["model"]
            history_df = model_info["history_df"]  # Get history_df from cache
            meta = model_info.get("meta", {})
            cached = True
        else:
            # Prepare prophet dataframe (includes regressors and ma)
            df_prophet, meta = prepare_prophet_dataframe(symbol, include_regressors=True)
            
            # Train prophet
            model = train_prophet(df_prophet, region=meta.get('region', 'US'))
            
            # Cache the model AND the history dataframe
            MODEL_CACHE["prophet"][symbol] = {
                "model": model,
                "history_df": df_prophet,  # Store history_df for predictions
                "trained_at": datetime.utcnow().isoformat(),
                "meta": meta
            }
            history_df = df_prophet
            cached = False

        # Predict - now passing all required arguments
        forecast_df = predict_prophet(model, history_df=history_df, periods=periods, freq='D')
        predictions = forecast_df.to_dict(orient="records")

        return jsonify({
            "symbol": symbol,
            "model": "prophet",
            "period_requested": period_key,
            "period_used_days": periods,
            "cached_model_used": cached,
            "trained_at": MODEL_CACHE["prophet"][symbol]["trained_at"],
            "meta": MODEL_CACHE["prophet"][symbol].get("meta", {}),
            "predictions": predictions
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Transformer route
# ---------------------------
@app.route('/predict/transformer', methods=['POST'])
def predict_transformer_route():
    try:
        payload = request.get_json(force=True)
        symbol = payload.get("symbol", "AAPL")
        period_key = payload.get("period", "7d")
        seq_length = int(payload.get("seq_length", 60))
        refresh = bool(payload.get("refresh", False))

        if period_key not in PERIOD_MAPPING:
            return jsonify({"error": "Invalid period. Choose one of: " + ", ".join(PERIOD_MAPPING.keys())}), 400

        requested_days = PERIOD_MAPPING[period_key]
        # cap transformer horizon
        n_days = min(requested_days, MAX_TRANSFORMER_STEPS)

        # Check cache
        cache_hit = (not refresh) and (symbol in MODEL_CACHE["transformer"])
        if cache_hit:
            model_info = MODEL_CACHE["transformer"][symbol]
            model = model_info["model"]
            scaler = model_info["scaler"]
            cached_seq_length = model_info.get("seq_length", 60)
            meta = model_info.get("meta", {})
            cached = True
            # If user requested a different seq_length than cached, we should retrain
            if cached_seq_length != seq_length:
                cache_hit = False
        if not cache_hit:
            # Prepare transformer data (no train/test split)
            X, y, scaler, df_full = prepare_transformer_dataframe(symbol, seq_length=seq_length)
            # Train transformer model
            model, history = train_transformer_model(X, y, seq_length=seq_length, n_features=X.shape[2])
            MODEL_CACHE["transformer"][symbol] = {
                "model": model,
                "scaler": scaler,
                "seq_length": seq_length,
                "trained_at": datetime.utcnow().isoformat(),
                "meta": {
                    "first_date": df_full['ds'].min().isoformat(),
                    "last_date": df_full['ds'].max().isoformat()
                }
            }
            cached = False
            meta = MODEL_CACHE["transformer"][symbol]["meta"]
        else:
            # cached model exists; need df to run predict_future (prepare_transformer_dataframe gives df)
            # We prepare df only (not re-scaling) so we can run predict_future which uses scaler
            _, _, _, df_full = prepare_transformer_dataframe(symbol, seq_length=seq_length)

        # Predict future n_days using recursive forecasting
        preds = predict_future(model, df_full, scaler, seq_length=seq_length, n_days=n_days)

        # Return results: list of {day_index:int, predicted_close:float}
        predictions = []
        for i, p in enumerate(preds, start=1):
            predictions.append({"day_index": i, "predicted_close": float(p)})

        return jsonify({
            "symbol": symbol,
            "model": "transformer",
            "period_requested": period_key,
            "period_used_days": n_days,
            "cached_model_used": cached,
            "trained_at": MODEL_CACHE["transformer"][symbol]["trained_at"],
            "meta": meta,
            "predictions": predictions
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------
# Cache info and management
# ---------------------------
@app.route('/cache', methods=['GET'])
def cache_info():
    return jsonify({
        "prophet_cached_symbols": list(MODEL_CACHE["prophet"].keys()),
        "transformer_cached_symbols": list(MODEL_CACHE["transformer"].keys())
    })


@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    payload = request.get_json(force=True) or {}
    symbol = payload.get("symbol", None)
    model_type = payload.get("model", None)  # "prophet" or "transformer"

    if model_type not in (None, "prophet", "transformer"):
        return jsonify({"error": "model must be 'prophet' or 'transformer'"}), 400

    if symbol:
        # clear single symbol
        if model_type:
            MODEL_CACHE[model_type].pop(symbol, None)
        else:
            MODEL_CACHE["prophet"].pop(symbol, None)
            MODEL_CACHE["transformer"].pop(symbol, None)
    else:
        # clear all or specific model_type
        if model_type == "prophet":
            MODEL_CACHE["prophet"].clear()
        elif model_type == "transformer":
            MODEL_CACHE["transformer"].clear()
        else:
            MODEL_CACHE["prophet"].clear()
            MODEL_CACHE["transformer"].clear()

    return jsonify({"status": "cache_cleared", "symbol": symbol, "model": model_type})


# ---------------------------
# 404 handler
# ---------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "endpoint not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
