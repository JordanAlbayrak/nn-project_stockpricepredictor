import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import yfinance as yf

# Load data
file_path = 'tesla_data_formatted.csv'
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])  # Convert the 'Date' column to datetime format

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
dataset_train = np.array(prices_scaled[:int(prices_scaled.shape[0] * 0.8)])
dataset_test = np.array(prices_scaled[int(prices_scaled.shape[0] * 0.8):])

x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)

# Reshape input for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

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
model.fit(x_train, y_train, epochs=50, batch_size=32)
model.save('stock_prediction.keras')

# Load the trained model
model = load_model('stock_prediction.keras')

# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the entire dataset
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor('#000041')

# Plot original prices
ax.plot(dates, prices, color='gray', label='Original prices')

# Plot predicted prices
ax.plot(dates[-len(predictions):], predictions, color='cyan', label='Predicted prices')

plt.legend()
plt.show()
