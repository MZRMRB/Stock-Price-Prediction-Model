project 04:

Stock Price Prediction
Goal: Predict stock prices using historical data.
Steps:

Dataset: Yahoo Finance stock data.
Model: Use LSTM or Random Forest Regression.
Evaluation: RMSE and directional accuracy.

https://www.kaggle.com/datasets/suruchiarora/yahoo-finance-dataset-2018-2023 

##About DataSet:

The "yahoo_finance_dataset(2018-2023)" dataset is a financial dataset containing daily stock market data for multiple assets such as equities, ETFs, and indexes. It spans from April 1, 2018 to March 31, 2023, and contains 1257 rows and 7 columns. The data was sourced from Yahoo Finance, and the purpose of the dataset is to provide researchers, analysts, and investors with a comprehensive dataset that they can use to analyze stock market trends, identify patterns, and develop investment strategies.

The dataset includes the following columns:

Date: The date on which the stock market data was recorded.
Open: The opening price of the asset on the given date.
High: The highest price of the asset on the given date.
Low: The lowest price of the asset on the given date.
Close: The closing price of the asset on the given date. Note that this price does not take into account any after-hours trading that may have occurred after the market officially closed. Adj Close*: The adjusted closing price of the asset on the given date. This price takes into account any dividends, stock splits, or other corporate actions that may have occurred, which can affect the stock price.
Volume: The total number of shares of the asset that were traded on the given date.



We’ve successfully loaded and inspected the Yahoo Finance dataset (Apr 2018–Mar 2023) and confirmed it has 1,258 trading days and six numeric fields (Open, High, Low, Close, Adj Close, Volume). Here’s how we can proceed:

---

### 1. Data Preprocessing & Feature Engineering
- **Date handling**: Already parsed & set as index so we can easily slice by time.
- **Missing values**: Check for any NaNs (unlikely here, but always good to verify).
- **Technical indicators** (optional, but often helpful):
  - Moving averages (e.g. 5-day, 20-day)
  - Bollinger Bands
  - Momentum or RSI
  - Volume-based features (e.g. volume change)
- **Lag features** for modelling:
  - Previous day’s Close (and possibly other prices/volumes)
  - Rolling-window statistics (mean, std)

### 2. Train/Test Split
- Perform a **time-series split**, e.g.:
  - Train on data up to end-2021 (≈80% of records)
  - Test on 2022–Mar 2023 (≈20%)

### 3. Random Forest Regression
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1) Select features (e.g., lag_1_Close, MA_5, Volume, …)
# 2) Split into X_train, y_train and X_test, y_test
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

rmse_rf = mean_squared_error(y_test, pred_rf, squared=False)
```
- **Hyperparameters to tune**: `n_estimators`, `max_depth`, `min_samples_split`

### 4. LSTM Model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 1) Scale numeric features to [0,1] with MinMaxScaler
# 2) Create sliding windows: e.g., last 20 days of features → predict next day’s Close
# 3) Build and compile:
model = Sequential([
    LSTM(50, input_shape=(window_size, num_features)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(train_X, train_y, epochs=50, batch_size=32, validation_split=0.1)
pred_lstm = model.predict(test_X)
rmse_lstm = np.sqrt(mean_squared_error(test_y, pred_lstm))
```
- **Key points**: sequence length (`window_size`), number of LSTM units, epochs, batch size.

### 5. Evaluation Metrics
- **RMSE** (Root Mean Squared Error) for absolute accuracy.
- **Directional Accuracy**:  
  \[
    \text{DirAcc} = \frac{\text{\# correct “up”/“down” predictions}}{\text{total predictions}}
  \]
  ```python
  direction_true = np.sign(y_test[1:] - y_test[:-1])
  direction_pred = np.sign(predictions[1:] - predictions[:-1])
  directional_accuracy = np.mean(direction_true == direction_pred)
  ```


COMMON QUESTIONS:
why is shifting, dropping NAs, splitting, reshaping, scaling necessary? especially why yesterday's close necessary to make?

Every step in that snippet serves to turn raw historical prices into the exact form your models need—and the single most important “feature” in a naive time-series model is simply “what happened yesterday?” Here’s why:

Why create yesterday’s close (the lag feature)?

Time-series are autocorrelated: stock prices don’t jump around completely at random; today’s price tends to be close to yesterday’s. By explicitly giving the model “Close_lag1” (yesterday’s Close), you give it its strongest predictive signal.

Machine-learning models need features: unlike classical econometric ARIMA models, tree-based regressors or neural nets don’t inherently know “use the prior value”—you must supply that as a column.

dropna() after shifting

The very first row loses its lagged value (there is no “day -1”), so you drop that one incomplete sample to keep every row fully populated.

80/20 train/test split

Prevent leakage: by slicing chronologically, the test set truly simulates “future” data the model has never seen.

Consistency: taking the first 80% for training and the last 20% for testing is a simple, reproducible rule for evaluation.

Flat vs. sequence inputs

Flat arrays (X_train = [[Close_lag1], …]) are exactly what non-sequence models (e.g. Random Forest) expect: one feature per row.

Sequences (make_seq) reshape your data into windows (here length 1) so that recurrent architectures (LSTM) get inputs of shape (batch, time_steps, features). Even with a window of 1, you still need that extra dimension.

Scaling & inverse-scaling

Neural nets train more reliably when inputs are scaled into a narrow range (e.g. [0, 1]).

To interpret errors in actual rupees or dollars, you must invert that scaling on the model’s outputs before computing RMSE.

In short:

Yesterday’s close is your single predictor because it’s the simplest and most powerful time-series feature—prices move in small increments, so “what it was yesterday” carries most of the information about “what it will be today.”

Every other line (shifting, dropping NAs, splitting, reshaping, scaling) is just plumbing to get that data into exactly the format each algorithm demands.



what is window size?

In time-series modelling, window size (sometimes called “look-back” or “sequence length”) is simply the number of prior time steps you feed into your model to predict the next value. Here’s how it works:

Basic idea

If your window size is w, you take the last w observations as a single input sample.

You then try to predict the observation immediately after those w points.

Example with window size = 1

Inputs:

sql
Copy
Edit
[day₀_close] → predict → day₁_close  
[day₁_close] → predict → day₂_close  
[day₂_close] → predict → day₃_close
…  
Each sample consists of exactly one previous close (“lag 1”), so your network sees shape (batch_size, 1 time-step, 1 feature).

Example with window size = 3

Inputs:

sql
Copy
Edit
[day₀, day₁, day₂] → predict → day₃  
[day₁, day₂, day₃] → predict → day₄  
[day₂, day₃, day₄] → predict → day₅  
…  
Now each sample is a 3-day history. Your LSTM input shape becomes (batch_size, 3 time-steps, 1 feature).

Why window size matters

Short windows (e.g. 1–5 days) capture only very recent trends—good if you believe only the last few days drive tomorrow.

Longer windows let the model learn from longer-term patterns (weekly, monthly), but they:

Increase the number of weights (and training time).

Require more data to avoid overfitting.

Implementation in code

python
Copy
Edit
def make_seq(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        # data[i-window_size : i] is a window of length `window_size`
        X.append(data[i-window_size : i, feature_index])
        y.append(data[i, feature_index])
    return np.array(X), np.array(y)
If window_size=1, each X[i] is just 1 value.

If window_size=3, each X[i] is an array of 3 consecutive values.

In short: window size tells your model “how many past days to look at” when making each prediction. It shapes the input data into fixed-length sequences, which recurrent or convolutional time-series models require.