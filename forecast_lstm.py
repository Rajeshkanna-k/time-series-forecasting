import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# 1. DATA GENERATION (Task 1)
def get_multivariate_data(n_obs=5200):
    np.random.seed(42)
    time = np.arange(n_obs)
    # Target: Trend + Seasonality + Noise
    y = 0.03 * time + 12 * np.sin(2 * np.pi * time / 365) + np.random.normal(0, 1.5, n_obs)
    # 4 Additional Features
    f1 = np.roll(y, 1) * 0.7 + np.random.normal(0, 0.5, n_obs)
    f2 = np.sin(2 * np.pi * time / 7) * 8
    f3 = np.random.uniform(15, 25, n_obs)
    f4 = np.cos(2 * np.pi * time / 365) * 5
    return pd.DataFrame({'y': y, 'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4}).fillna(0)

df = get_multivariate_data()

# 2. PREPROCESSING
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

def window_data(data, window=24):
    x, target = [], []
    for i in range(len(data) - window):
        x.append(data[i:i+window])
        target.append(data[i+window, 0])
    return np.array(x), np.array(target)

X, y = window_data(scaled)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. LSTM MODEL (Task 2)
model = Sequential([
    LSTM(50, activation='relu', input_shape=(24, 5), return_sequences=True),
    Dropout(0.2),
    LSTM(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# 4. EXPLAINABILITY (Task 3)
# Extracting weights as a proxy for feature importance
weights = np.mean(np.abs(model.layers[0].get_weights()[0]), axis=1)
importance = dict(zip(['Lag', 'Seasonality', 'Random', 'Annual', 'External'], weights))

# 5. METRICS (Task 4)
preds = model.predict(X_test, verbose=0)
rmse = np.sqrt(np.mean((y_test - preds.flatten())**2))
# Naive Baseline (Persistence)
baseline_rmse = np.sqrt(np.mean((y_test[1:] - y_test[:-1])**2))
