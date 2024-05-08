import os
import yfinance as yf

"""
https://medium.com/plumbersofdatascience/extracting-yahoo-finance-stock-data-building-a-simple-etl-script-82bf645cff3c
종가(close)는 시장이 마감되기 전 마지막으로 거래된 주가를 뜻하는데 수정 종가(adjusted close)는 해당 주식의 종가(close)에 분할(splits), 
배당금 분배(dividend distributions) 등 주가에 영향을 미칠 수 있는 기업의 활동(corporate actions)을 반영한 후의 종가
data_frame['% Change'] = round(data_frame['Adj Close'] / data_frame['Adj Close'].shift(1) - 1, 4)
"""
import pandas as pd
import os
import yfinance as yf

# Downloading the stock prices data
TICKER = 'AAPL'
START_DATE = '2022-06-01'
END_DATE = '2024-05-01'
INTERVAL = '1h'
BREAKPOINT_DATE = '2023-06-01'

stock_prices_df = yf.download(TICKER, start=START_DATE, end=END_DATE, interval=INTERVAL)
stock_prices_df.reset_index(inplace=True)

# Add a new column 'ticker' with the value 'AAPL' at the first position (index 0)
stock_prices_df.insert(0, 'Ticker', TICKER)

print(stock_prices_df)
stock_prices_df.rename(columns={"Adj Close": "AdjClose"}, inplace=True)
print(stock_prices_df)

# stock_prices_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yfdata.csv'), index=False)

# Splitting the DataFrame based on the date condition
train_df = stock_prices_df[stock_prices_df['Datetime'] < BREAKPOINT_DATE]
stream_df = stock_prices_df[stock_prices_df['Datetime'] >= BREAKPOINT_DATE]

# producing outlier to be filtered out by KStream later
for i in range(99, len(stream_df), 100):
    stream_df.iloc[i, stream_df.columns.get_loc('AdjClose')] *= 10

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct file paths
train_data_csv = os.path.join(script_dir, 'train_yfdata.csv')
stream_data_csv = os.path.join(script_dir, 'stream_yfdata.csv')

# Save the DataFrames to CSV
train_df.to_csv(train_data_csv, index=False)
stream_df.to_csv(stream_data_csv, index=False)

# Print column names
print(stock_prices_df.columns)
