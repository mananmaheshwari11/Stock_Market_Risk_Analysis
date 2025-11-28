import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# TRANSFORMER MODEL ARCHITECTURE
# ==========================================

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Single Transformer Encoder Block"""
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Network
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def build_transformer_model(
    seq_length,
    n_features,
    head_size=256,
    num_heads=8,
    ff_dim=128,
    num_transformer_blocks=3,
    mlp_units=[128, 64],
    dropout=0.15,
    mlp_dropout=0.2,
    learning_rate=0.0005
):
    """Builds the Transformer Model"""
    inputs = keras.Input(shape=(seq_length, n_features))
    x = inputs

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='huber',
        metrics=['mae']
    )
    return model

# ==========================================
# TRAINING FUNCTION
# ==========================================

def train_transformer_model(X, y, seq_length, n_features, epochs=150, batch_size=16):
    """
    Train the Transformer on given data.
    Returns trained model and training history.
    """
    model = build_transformer_model(seq_length, n_features)
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, verbose=0
    )

    history = model.fit(
        X, y,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    return model, history

# ==========================================
# FUTURE PREDICTION (N-DAY FORECAST)
# ==========================================

def predict_future(model, df, scaler, seq_length, n_days=7):
    """
    Predicts next N days closing prices using recursive forecasting.
    
    Parameters:
      model     : trained transformer model
      df        : full historical dataframe with features
      scaler    : fitted MinMaxScaler
      seq_length: number of timesteps used in sequence
      n_days    : how many days ahead to predict

    Returns:
      - np.array of predicted close prices (inverse-transformed)
    """
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns',
                    'MA_7', 'MA_21', 'Volatility', 'Volume_MA']
    
    data_features = df[feature_cols].values.astype(float)
    scaled_data = scaler.transform(data_features)

    # start with last seq_length window
    last_sequence = scaled_data[-seq_length:].copy()
    predictions_scaled = []

    for _ in range(n_days):
        seq_input = np.expand_dims(last_sequence, axis=0)  # shape (1, seq_len, n_features)
        next_pred_scaled = model.predict(seq_input, verbose=0)[0][0]
        predictions_scaled.append(next_pred_scaled)

        # append predicted Close into next sequence
        new_row = last_sequence[-1].copy()
        new_row[3] = next_pred_scaled  # update Close value
        last_sequence = np.vstack([last_sequence[1:], new_row])

    # inverse transform predicted Close values
    dummy = np.zeros((len(predictions_scaled), scaled_data.shape[1]))
    dummy[:, 3] = predictions_scaled
    inv_predictions = scaler.inverse_transform(dummy)[:, 3]

    return inv_predictions
