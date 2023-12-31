import yfinance as yf
import pandas as pd

# Define the ticker symbol
ticker = "META"

# Define the time range
start_date = pd.to_datetime("2023-01-01")
end_date = pd.to_datetime("2023-10-01")

# Download historical stock data using yfinance
data = yf.download(ticker, start=start_date, end=end_date)

# Save the data to a CSV file
csv_filename = f"{ticker}_daily_2023-01-01_2023-10-01_TEST.csv"
data.to_csv(csv_filename)

print(f"Stock data saved to {csv_filename}")