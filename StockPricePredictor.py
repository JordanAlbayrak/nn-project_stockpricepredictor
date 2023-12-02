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
df['Date'] = pd.to_datetime(df['Date'], utc=True)  # Convert the 'Date' column to datetime format

# Extract 'Open' prices and dates
prices = df['Open'].values
dates = df['Date'].values

# Reshape 'Open' prices
prices = prices.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)


# Create dataset
def create_dataset(df):
    x = []
    y = []
    for i in range(50, df.shape[0]):
        x.append(df[i - 50:i, 0])
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x, y


# Train-test split
dataset_train = np.array(prices_scaled)

x_train, y_train = create_dataset(dataset_train)

# Reshape input for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

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
hist = model.fit(x_train, y_train, epochs=21, batch_size=4, verbose=2)
plt.plot(hist.history['loss'])
plt.title('Training model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()
model.save('stock_prediction.keras')

# Load the trained model
model = load_model('stock_prediction.keras')

# Make predictions
file_path = 'AAPL_daily_2023-01-01_2023-10-01_TEST.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'], utc=True)  # Convert the 'Date' column to datetime format

# Extract 'Open' prices and dates
prices = df['Open'].values
dates = df['Date'].values

# Reshape 'Open' prices
prices = prices.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)
dataset_test = np.array(prices_scaled)
x_test, y_test = create_dataset(dataset_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Future Prediction
x_extended = x_test[-1]
num_predictions = 30

future_predictions = []
for _ in range(num_predictions):
    prediction = model.predict(x_extended.reshape(1, x_extended.shape[0], x_extended.shape[1]))
    future_predictions.append(prediction[0, 0])
    x_extended = np.roll(x_extended, -1)
    x_extended[-1] = prediction[0, 0]

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Get the last date in the original data
last_date = df['Date'].iloc[-1]

# Generate future dates starting from the next business day
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=0), periods=num_predictions, freq='B')

# Plot the entire dataset
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor('#000041')

# Plot original prices
ax.plot(dates, prices, color='gray', label='Original prices')

# Plot predicted prices
ax.plot(dates[len(dates) - predictions.shape[0]:], predictions, color='cyan', label='Predicted prices')

# Plot future predictions
ax.plot(future_dates, future_predictions, color='magenta', linestyle='dashed', label='Future predictions')

plt.legend()
plt.show()
