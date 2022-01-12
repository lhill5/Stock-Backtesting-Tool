# uses polygon.io API to get data (no-usable, have to pay for license)

import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, date, timedelta

api_key = '8DSUXEKEJXDQTA6F'

ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')

stock_dict = data.to_dict()
for key in list(stock_dict.keys()):
    new_key = key[3:]
    stock_dict[new_key] = stock_dict.pop(key)

# start = datetime(2021,5,26,9,0,0)
# end = datetime(2021,5,26,10,0,0)
end = date.today()
start = end - timedelta(days=10)

time_data = list(pd.date_range(start,end,freq='d'))
print(time_data)

time_series = None
for key, stock in stock_dict.items():
    if key != 'close':
        continue

    last_date_data = list(stock.items())[0]

print(f'date: {last_date_data[0]}, price: {last_date_data[1]}')



