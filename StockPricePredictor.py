import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import yfinance as yf

# Load data
file_path = 'AAPL_daily_2017-01-01_2023-01-01.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df['Formatted_Date'] = df['Date'].dt.strftime('%Y-%d-%m')

# Extract 'Open' prices and dates
prices = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values
dates = df['Formatted_Date'].values

# Reshape 'Open' prices
print("test",prices.shape)
prices = prices[:, list(range(0,prices.shape[-1]))].reshape(-1, prices.shape[-1])
print('prices',prices.shape)
# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices[:, list(range(0,prices.shape[-1]))])
print(prices_scaled)


y_train = prices_scaled[1:int(prices_scaled.shape[0] * 0.8) + 1]
x_train = prices_scaled[:int(prices_scaled.shape[0] * 0.8)]

y_test = prices_scaled[int(prices_scaled.shape[0] * 0.8) + 1:int(prices_scaled.shape[0] * 0.9) + 1]
x_test = prices_scaled[int(prices_scaled.shape[0] * 0.8):int(prices_scaled.shape[0] * 0.9)]

y_validation = prices_scaled[int(prices_scaled.shape[0] * 0.90)+1:-1]
x_validation = prices_scaled[int(prices_scaled.shape[0] * 0.90):-2]

# Reshape input for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], prices.shape[-1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], prices.shape[-1], 1))

# Build LSTM model
# model = Sequential()
# model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=96, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=96, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=96))
# model.add(Dropout(0.2))
# model.add(Dense(units=1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=21, batch_size=32, validation_data=(x_validation, y_validation))
# model.save('stock_prediction.keras')

# Load the trained model
model = load_model('stock_prediction.keras')

# Generate predictions for the given dataset
print("x",x_test.shape)
print("y",y_test.shape)
print("reshape", np.reshape(x_test, (x_test.shape[0], y_test.shape[-1], 1)).shape)
predictions = model.predict(np.reshape(x_test, (x_test.shape[0], y_test.shape[-1], 1)))
print("prediction",predictions.shape)
predictions = scaler.inverse_transform(predictions)
# Generate future predictions
x_extended = x_test[-1]
num_predictions = 30

future_predictions = []
for _ in range(num_predictions):
    print(x_extended)
    prediction = model.predict(x_extended.reshape(1, x_extended.shape[0], x_extended.shape[1]))
    future_predictions.append(prediction[0, 0])
    x_extended = np.roll(x_extended, -1)
    x_extended[-1] = prediction[0, 0]

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Get the last date in the original data
last_date = df['Date'].iloc[int(df.shape[0] * 0.8):int(df.shape[0] * 0.9)].iloc[-1]

# Generate future dates starting from the next business day
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=0), periods=num_predictions, freq='B')

# Plot the entire dataset and future predictions
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor('#000041')

# Convert indices to integers
indices = df.index[int(df.shape[0] * 0.8):int(df.shape[0] * 0.9)].astype(int)

# Plot original prices
ax.plot(df['Date'].iloc[indices].values, df['Open'].iloc[indices].values, color='gray', label='Original prices')

# Plot predicted prices on the test data
ax.plot(df['Date'].iloc[indices].values, predictions, color='cyan', label='Predicted prices (Test Data)')

# Plot future predictions
ax.plot(future_dates, future_predictions, color='magenta', linestyle='dashed', label='Future predictions')

plt.legend()
plt.show()
