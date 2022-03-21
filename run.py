from stock_package.backtrader.SQL_DB import SQL_DB
from stock_package.backtrader.Graph import Graph
from stock_package.backtrader.Stock import Stock
from stock_package.backtrader.Stock_Prices import StockData_API
import stock_package.backtrader.financial_calcs

import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_stock_data(stock):
    stock_obj = Stock(stock, database, start_date, end_date)
    return stock_obj


def store_data(data):
    global stocks, stock_count

    stock, stock_obj = data
    stock_count += 1
    print(f'{stock:5s} - stock count: {stock_count}')
    stocks[stock] = stock_obj


stocks = {}
stock_count = 0

stock_data_API = StockData_API()
database = SQL_DB(stock_data_API, update=False)

start_date, end_date = datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)
stock = Stock('aapl', database, start_date=start_date, end_date=end_date)

plt.plot(stock.dates, stock.prices['high'])
plt.show()

