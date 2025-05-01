# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- Page config ---
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# --- Sidebar: Model & Data Inputs ---
st.sidebar.header("Model Configuration")
n_estimators = st.sidebar.slider("RF: Number of Trees", 10, 500, 100, step=10)
window_size = st.sidebar.slider("LSTM: Window Size", 1, 10, 1)
units = st.sidebar.slider("LSTM: Units", 1, 100, 10)
epochs = st.sidebar.slider("LSTM: Epochs", 1, 100, 20)
train_split = st.sidebar.slider("Train/Test Split (%)", 50, 90, 80)

st.sidebar.markdown("---")
st.sidebar.header("Your Today's Data")
input_open = st.sidebar.number_input("Open", min_value=0.0, step=0.01)
input_high = st.sidebar.number_input("High", min_value=0.0, step=0.01)
input_low = st.sidebar.number_input("Low", min_value=0.0, step=0.01)
input_close = st.sidebar.number_input("Close", min_value=0.0, step=0.01)
input_adj_close = st.sidebar.number_input("Adj Close", min_value=0.0, step=0.01)
input_volume = st.sidebar.number_input("Volume", min_value=0.0, step=1.0)

st.sidebar.markdown("---")
run = st.sidebar.button("Run Prediction 🚀")

if run:
    # 1. Load historical data
    @st.cache
    def load_data(path):
        df = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
        return df

    df = load_data('yahoo_data.csv')

    # 2. Clean & select closing price column
    close_col = 'Close*' if 'Close*' in df.columns else 'Close'
    df = df[[close_col]].dropna()
    df[close_col] = df[close_col].astype(str).str.replace(',', '').astype(float)

    # 3. Add lag feature
    df['Close_lag1'] = df[close_col].shift(1)
    df.dropna(inplace=True)

    # 4. Train/test split
    split_idx = int(len(df) * (train_split / 100))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    # 5. Prepare flat data for RF
    X_train = train_df[['Close_lag1']].values
    y_train = train_df[close_col].values
    X_test = test_df[['Close_lag1']].values
    y_test = test_df[close_col].values

    # 6. Train RF
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
    dir_rf = np.mean(np.sign(y_test[1:] - y_test[:-1]) == np.sign(pred_rf[1:] - pred_rf[:-1]))

    # 7. Prepare sequences for LSTM
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[close_col]].values)
    def make_seq(data, w):
        X, y = [], []
        for i in range(w, len(data)):
            X.append(data[i-w:i, 0:1])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    X_seq, y_seq = make_seq(scaled, window_size)
    X_tr, X_te = X_seq[:split_idx-window_size], X_seq[split_idx-window_size:]
    y_tr, y_te = y_seq[:split_idx-window_size], y_seq[split_idx-window_size:]

    # 8. Train LSTM
    model = Sequential([
        LSTM(units, input_shape=(window_size,1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_tr, y_tr, epochs=epochs, batch_size=8, verbose=0)
    pred_scl = model.predict(X_te).flatten()
    y_true = scaler.inverse_transform(y_te.reshape(-1,1)).flatten()
    y_pred = scaler.inverse_transform(pred_scl.reshape(-1,1)).flatten()
    rmse_lstm = np.sqrt(mean_squared_error(y_true, y_pred))
    dir_lstm = np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1]))

    # --- Display historical model performance ---
    st.markdown("## 🚀 Historical Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Random Forest")
        st.metric("RMSE", f"{rmse_rf:.2f}")
        st.metric("Dir. Accuracy", f"{dir_rf:.1%}")
    with col2:
        st.subheader("LSTM")
        st.metric("RMSE", f"{rmse_lstm:.2f}")
        st.metric("Dir. Accuracy", f"{dir_lstm:.1%}")

    # 9. Forecast next day from historical last close
    last_close = df[close_col].iloc[-1]
    rf_next_hist = rf.predict([[last_close]])[0]
    lstm_seq_hist = scaled[-window_size:].reshape(1, window_size, 1)
    lstm_next_hist = scaler.inverse_transform(model.predict(lstm_seq_hist).reshape(-1,1))[0,0]
    dir_rf_hist = "📈 Up" if rf_next_hist > last_close else "📉 Down"
    dir_lstm_hist = "📈 Up" if lstm_next_hist > last_close else "📉 Down"

    st.markdown("---")
    st.markdown("### 📅 Forecast Based on Last Historical Close")
    st.write(f"As of {df.index[-1].date()}: Last Close = {last_close:.2f}")
    st.write(f"- RF predicts: {dir_rf_hist} (≈ {rf_next_hist:.2f})")
    st.write(f"- LSTM predicts: {dir_lstm_hist} (≈ {lstm_next_hist:.2f})")

    # 10. User-driven input forecast
    st.markdown("---")
    st.markdown("## ✍️ Forecast Based on Your Input")
    st.write("Enter today’s OHLC, Adj Close, and Volume, then hit **Run Prediction 🚀** again.")

    # Use input_close as yesterday's close
    if input_close > 0:
        # RF forecast
        rf_next = rf.predict([[input_close]])[0]
        dir_rf = "📈 Up" if rf_next > input_close else "📉 Down"
        # LSTM forecast (window_size=1 uses only Close)
        inp_scaled = scaler.transform([[input_close]])
        lstm_inp = inp_scaled.reshape(1, window_size, 1)
        lstm_next = scaler.inverse_transform(model.predict(lstm_inp).reshape(-1,1))[0,0]
        dir_lstm = "📈 Up" if lstm_next > input_close else "📉 Down"

        st.write(f"- RF based on your Close={input_close:.2f}: {dir_rf} (≈ {rf_next:.2f})")
        st.write(f"- LSTM based on your Close={input_close:.2f}: {dir_lstm} (≈ {lstm_next:.2f})")
    else:
        st.info("Please enter a valid Close price to get your forecast.")

    # 11. Comparison chart (aligned lengths)
    st.markdown("---")
    st.line_chart(pd.DataFrame({
        "Actual": y_true,
        "RF_Pred": pred_rf,
        "LSTM_Pred": y_pred
    }, index=test_df.index))

else:
    st.title("📊 Welcome to the Stock Predictor")
    st.write("Use the sidebar to configure models, input your data, and click **Run Prediction 🚀**.")
