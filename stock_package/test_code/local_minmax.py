from stock_package.backtrader.SQL_DB import SQL_DB
from stock_package.backtrader.Graph import Graph
from stock_package.backtrader.Stock import Stock
from stock_package.backtrader.StockData_API import StockData_API
import stock_package.backtrader.financial_calcs

import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_minmax_manual(stock):
    local_max_dates = []
    local_min_dates = []
    local_max_prices = []
    local_min_prices = []

    dates = stock.dates
    prices = stock.prices['high']

    for i in range(1, len(stock.dates)-1):
        if prices[i-1] < prices[i] > prices[i+1]:
            local_max_dates.append(dates[i])
            local_max_prices.append(prices[i])
        elif prices[i-1] > prices[i] < prices[i+1]:
            local_min_dates.append(dates[i])
            local_min_prices.append(prices[i])

    plt.plot(dates, prices)
    plt.scatter(local_min_dates, local_min_prices, c='g')
    plt.scatter(local_max_dates, local_max_prices, c='r')
    plt.show()


if __name__ == '__main__':
    stock_data_API = StockData_API()
    database = SQL_DB(stock_data_API, update=False)

    start_date, end_date = datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)
    stock = Stock('tsla', database, start_date=start_date, end_date=end_date)

    get_minmax_manual(stock)

