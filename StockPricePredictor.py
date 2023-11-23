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
print(df.info())
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df['Formatted_Date'] = df['Date'].dt.strftime('%Y-%d-%m')

# Extract 'Open' prices and dates
prices = df[['Open','High', 'Low', 'Close', 'Adj Close', 'Volume']].values
dates = df['Formatted_Date'].values

# Reshape 'Open' prices
prices = prices[:, 3].reshape(-1, 1)
# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)
print(prices_scaled.shape)
# print("prices_scaled",prices_scaled)


y_train = []
x_train = []

for i in range (60, prices_scaled.shape[0]):
    x_train.append(prices_scaled[i-60:i,0])
    y_train.append(prices_scaled[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

print(y_train.shape)
print(x_train.shape)




# Reshape input for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print("x_train",x_train.shape)


# Build LSTM model
model = Sequential()
model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=35, batch_size=32, verbose=2)
plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'],loc='upper right')
plt.show()
model.save('stock_prediction.keras')

# Load the trained model
model = load_model('stock_prediction.keras')

# Generate predictions for the given dataset
# Load test data
file_path = 'AAPL_daily_2023-01-01_2023-10-01_TEST.csv'
df = pd.read_csv(file_path)
print(df.info())
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df['Formatted_Date'] = df['Date'].dt.strftime('%Y-%d-%m')

# Extract 'Open' prices and dates
test_prices = df[['Open','High', 'Low', 'Close', 'Adj Close', 'Volume']].values
test_dates = df['Formatted_Date'].values

test_prices = test_prices[:, 3].reshape(-1, 1)
y_test = test_prices[60:,0:]

input_closing = test_prices[:,0:]

# Normalize data
scaler_test = MinMaxScaler(feature_range=(0, 1))
prices_scaled_test = scaler_test.fit_transform(input_closing)
print("test shape",prices_scaled_test.shape)

x_test = []
dates_array = []
for i in range(60,len(test_prices)):
    x_test.append(prices_scaled_test[i-60:i,0])
    dates_array.append(test_dates[i-60])
x_test = np.array(x_test)
dates_array = np.array(dates_array)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
print("X test shape",x_test.shape)
print("dates",dates_array.shape)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
# print("prediction",predictions)
# Generate future predictions
x_extended = x_test[-1]
num_predictions = 30

future_predictions = []
for _ in range(num_predictions):
    # print(x_extended)
    prediction = model.predict(x_extended.reshape(1, x_extended.shape[0], x_extended.shape[1]))
    future_predictions.append(prediction[0, 0])
    x_extended = np.roll(x_extended, -1)
    x_extended[-1] = prediction[0, 0]

future_predictions = np.array(future_predictions).reshape(-1, 1)
print("future shape",future_predictions.shape)

# Get the last date in the original data
last_date = df['Date'].iloc[-1]

# Generate future dates starting from the next business day
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=0), periods=num_predictions, freq='B')

# Plot the entire dataset and future predictions
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor('#000041')

# Convert indices to integers
indices = df.index.astype(int)

# Plot original prices
ax.plot(dates_array, y_test, color='gray', label='Original prices')

# Plot predicted prices on the test data
print("dates shape",dates_array)
print("predictions shape",predictions.shape)
ax.plot(dates_array, predictions, color='cyan', label='Predicted prices (Test Data)')

# Plot future predictions
# ax.plot(future_dates, future_predictions, color='magenta', linestyle='dashed', label='Future predictions')

plt.legend()
plt.show()
