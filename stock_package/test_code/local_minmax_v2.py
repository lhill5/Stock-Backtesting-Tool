from stock_package.backtrader.SQL_DB import SQL_DB
from stock_package.backtrader.Stock import Stock
from stock_package.backtrader.StockData_API import StockData_API
import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
import math


def get_minmax(stock):
    dates = stock.dates
    prices = stock.prices['high']

    df = pd.DataFrame(prices, columns=['data'])
    n = 2  # number of points to be checked before and after

    # Find local peaks
    df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                        order=n)[0]]['data']
    df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                        order=n)[0]]['data']
    df = df.rename(columns={'min': 'buy', 'max': 'sell'})
    return df


def plot_results(df):
    plt.scatter(df.index, df['buy'], c='g')
    plt.scatter(df.index, df['sell'], c='r')
    plt.plot(df.index, df['data'])
    plt.show()


def evaluate_strategy(indices):
    shares = 0
    total_invested = 0
    total_transactions = 0
    total_profit = 0
    cum_percent_profit = 0
    cum_positive_percent_profit = 0
    buy_queue = []

    for i, row in indices.iterrows():
        data, buy_price, sell_price = row
        # buy signal
        if not math.isnan(buy_price):
            buy_queue.append(buy_price)
            total_invested += buy_price
            shares += 1

        # sell signal
        elif not math.isnan(sell_price):
            if shares == 0:
                continue
            else:
                # sell all shares
                for p in buy_queue:
                    total_profit += sell_price - p
                    percent_profit = ((sell_price - p) / p) * 100
                    cum_percent_profit += percent_profit
                    if percent_profit > 0:
                        cum_positive_percent_profit += percent_profit

                    shares -= 1
                    total_transactions += 1
                buy_queue = []

                assert(shares == 0)

    buy_sell_gain = ((indices.iloc[-1]['data'] - indices.iloc[0]['data']) / (indices.iloc[0]['data'])) * 100

    print(f'total profit: {total_profit}')
    print(f'cumulative % gain: {cum_percent_profit}')
    print(f'positive cumulative % gain: {cum_positive_percent_profit}')
    print(f'buy_and_sell % gain: {buy_sell_gain}')


if __name__ == '__main__':
    stock_data_API = StockData_API()
    database = SQL_DB(stock_data_API, update=False)

    start_date, end_date = datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)
    stock = Stock('msft', database, start_date, end_date)

    minmax_indices = get_minmax(stock)
    print(minmax_indices)

    eval = evaluate_strategy(minmax_indices)
    # print_eval(eval)

    plot_results(minmax_indices)

