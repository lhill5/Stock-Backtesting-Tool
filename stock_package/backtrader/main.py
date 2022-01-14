# To run application:
#   bokeh serve --show backtrader

from Stock import Stock
from Graph import Graph
from SQL_DB import SQL_DB
from StockData_API import StockData_API
import trading_strategies as strat

import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.linear_model import LinearRegression
import random


stocks = {}
stock_count = 0

stock_data_API = StockData_API()
database = SQL_DB(stock_data_API, update=False)

start_time = time.time()
start_date, end_date = datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)
# stock_list = ['aapl', 'tsla', 'msft', 'voo', 'vti', 'gme', 'vbr', 'ijr', 'vo', 'amc']
stock_list = database.stocks
# for i in range(100):
for ticker in ['AAPL']:
# for ticker in database.stocks:
#     ticker = random.choice(database.stocks)
    # ticker = 'aapl'

    stock = Stock(ticker, database, start_date=start_date, end_date=end_date)
    stocks[ticker] = stock

    if stock.error:
        print('error')
        continue

    df, df_trades = strat.buysell_minmax(stock)
    eval = strat.evaluate_strategy(stock, df)

    # if ticker == 'MGM':
    # trading_strategies.plot_results(df, df_trades)

ticker = list(stocks.keys())[0]
fig = Graph(stocks[ticker], stocks, database.stocks, database)
fig.plot()

end_time = time.time()
print(f'program finished in {round(end_time - start_time, 1)} seconds')









# ___________________________________________________________________________________________ #

# breakpoint()
# end_time = time.time()
# print(f'program finished in {round(end_time - start_time, 1)} seconds')


# def get_stock_data(stock):
#     stock_obj = Stock(stock, database, start_date, end_date)
#     return stock_obj
#
#
# def store_data(data):
#     global stocks, stock_count
#
#     stock, stock_obj = data
#     stock_count += 1
#     print(f'{stock:5s} - stock count: {stock_count}')
#     stocks[stock] = stock_obj


# pool = ThreadPool(cpu_count())
# results = pool.map(get_stock_data, database.stocks)

# pool = Pool(cpu_count())
# for stock in database.stocks:
#     pool.apply_async(get_stock_data, args=(stock), callback=store_data)
# pool.close()
# pool.join()
