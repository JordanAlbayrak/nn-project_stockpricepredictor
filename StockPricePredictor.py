import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

# Constants
TRAIN_FILE_PATH = 'META_daily_2017-01-01_2023-01-01.csv'
TEST_FILE_PATH = 'META_daily_2023-01-01_2023-10-01_TEST.csv'
MODEL_FILE_PATH = 'stock_prediction.keras'
WINDOW_SIZE = 50
LSTM_UNITS = 96
EPOCHS = 21
BATCH_SIZE = 4
NUM_PREDICTIONS = 50


# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    prices = df['Open'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices)
    return df['Date'].values, prices, prices_scaled, scaler


# Function to create dataset for LSTM
def create_lstm_dataset(data):
    x, y = [], []
    for i in range(WINDOW_SIZE, len(data)):
        x.append(data[i - WINDOW_SIZE:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=LSTM_UNITS, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=LSTM_UNITS, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=LSTM_UNITS, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=LSTM_UNITS))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Load and preprocess training data
dates_train, prices_train, prices_scaled_train, scaler = load_and_preprocess_data(TRAIN_FILE_PATH)
x_train, y_train = create_lstm_dataset(prices_scaled_train)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

# Build and train the model
model = build_lstm_model((WINDOW_SIZE, 1))
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
model.save(MODEL_FILE_PATH)

# Plot training loss
plt.plot(history.history['loss'])
plt.title('Training Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()

# Load and preprocess test data
dates_test, prices_test, prices_scaled_test, _ = load_and_preprocess_data(TEST_FILE_PATH)
x_test, _ = create_lstm_dataset(prices_scaled_test)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

# Load the model and make predictions
model = load_model(MODEL_FILE_PATH)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Future predictions
x_extended = np.copy(x_test[-1])
future_predictions = []
for _ in range(NUM_PREDICTIONS):
    prediction = model.predict(x_extended.reshape(1, WINDOW_SIZE, 1))[0, 0]
    future_predictions.append(prediction)
    x_extended = np.roll(x_extended, -1)
    x_extended[-1] = prediction
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Plot results
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_facecolor('#000041')
ax.plot(dates_test, prices_test, color='gray', label='Original Prices')
ax.plot(dates_test[len(dates_test) - len(predictions):], predictions, color='cyan', label='Predicted Prices')
future_dates = pd.date_range(start=dates_test[-1], periods=NUM_PREDICTIONS, freq='B')
ax.plot(future_dates, future_predictions, color='magenta', linestyle='dashed', label='Future Predictions')
plt.legend()
plt.show()
