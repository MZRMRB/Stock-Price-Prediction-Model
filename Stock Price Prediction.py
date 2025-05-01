# 0. Imports
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Load & clean the dataset
df = pd.read_csv('yahoo_data.csv', parse_dates=['Date'], date_format='%Y-%m-%d', index_col='Date')

# Use correct 'Close' column name (Yahoo may name it 'Close*')
close_col = 'Close*' if 'Close*' in df else 'Close'

# Keep only the closing prices and drop missing values
df = df[[close_col]].dropna()

# Convert string-formatted prices to float
df[close_col] = df[close_col].astype(str).str.replace(',', '').astype(float)

# 2. Feature engineering: Add 1-day lag of closing price
df['Close_lag1'] = df[close_col].shift(1)
df.dropna(inplace=True)

# 3. Train/test split: 80% train, 20% test
split_idx = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

# 4. Prepare input and target arrays for ML models
X_train = train_df[['Close_lag1']].values
y_train = train_df[close_col].values
X_test  = test_df [['Close_lag1']].values
y_test  = test_df [close_col].values

# 5. Function to compute directional accuracy (up/down movement match)
def directional_accuracy(true, pred):
    true_dir = np.sign(true[1:] - true[:-1])
    pred_dir = np.sign(pred[1:] - pred[:-1])
    return np.mean(true_dir == pred_dir)

# --- Random Forest Regressor tuning ---
best_rf = None
best_rf_score = np.inf
best_rf_params = None

# Try different numbers of trees (n_estimators)
for n in [50, 100, 200]:
    rf = RandomForestRegressor(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    p = rf.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, p))
    dacc = directional_accuracy(y_test, p)
    
    print(f"RF n_estimators={n} → RMSE={rmse:.1f}, DirAcc={dacc:.3f}")
    
    if rmse < best_rf_score:
        best_rf_score = rmse
        best_rf = rf
        best_rf_params = {'n_estimators': n}

print("\nBest RF:", best_rf_params, f"RMSE={best_rf_score:.1f}\n")

# --- Prepare data for LSTM ---
scaler = MinMaxScaler()

# Scale the closing prices between 0 and 1
scaled = scaler.fit_transform(df[[close_col]].values)

# Function to create sequences for LSTM (window_size=1 for 1-day lag)
def make_seq(data, window_size=1):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0:1])  # shape: (window_size, 1)
        y.append(data[i, 0])                  # shape: ()
    return np.array(X), np.array(y)

# Generate sequences and split into train/test
X_seq, y_seq = make_seq(scaled, window_size=1)
X_train_seq = X_seq[:split_idx-1]
X_test_seq  = X_seq[split_idx-1:]
y_train_seq = y_seq[:split_idx-1]
y_test_seq  = y_seq[ split_idx-1:]

# --- LSTM tuning ---
best_lstm = None
best_lstm_score = np.inf
best_lstm_params = None

# Try different combinations of LSTM units and epochs
for units in [5, 10, 20]:
    for epochs in [10, 20]:
        model = Sequential([
            LSTM(units, input_shape=(1,1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=8, verbose=0)
        
        # Predict and inverse-transform the results to original scale
        pred_scl = model.predict(X_test_seq).flatten()
        y_true = scaler.inverse_transform(y_test_seq.reshape(-1,1)).flatten()
        y_pred = scaler.inverse_transform(pred_scl.reshape(-1,1)).flatten()
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        dacc = directional_accuracy(y_true, y_pred)
        
        print(f"LSTM units={units}, epochs={epochs} → RMSE={rmse:.1f}, DirAcc={dacc:.3f}")
        
        if rmse < best_lstm_score:
            best_lstm_score = rmse
            best_lstm = model
            best_lstm_params = {'units': units, 'epochs': epochs}

print("\nBest LSTM:", best_lstm_params, f"RMSE={best_lstm_score:.1f}")
