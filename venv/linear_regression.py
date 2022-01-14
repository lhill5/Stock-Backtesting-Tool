from Stock import Stock
from Graph import Graph
from SQL_DB import SQL_DB
import time
import datetime
import pandas as pd
from StockData_API import StockData_API
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from sklearn.linear_model import LinearRegression
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

# start_time = time.time()

stock = Stock('aapl', database, start_date=start_date, end_date=end_date)
# fig = Graph(stock, database.stocks)
# fig.plot()

# breakpoint()
# end_time = time.time()
# print(f'program finished in {round(end_time - start_time, 1)} seconds')

model = LinearRegression()
x_dates = stock.dates
x_ints = [i for i in range(len(x_dates))]

x = np.array(x_ints).reshape((-1, 1))
y = np.array(stock.prices['high'])

plt.plot(x_ints, y)

model.fit(x, y)
intercept = model.intercept_
slope = model.coef_

linear_reg_line = [slope*x_0 + intercept for x_0 in x_ints]
plt.plot(x_ints, linear_reg_line)
plt.show()

# pool = ThreadPool(cpu_count())
# results = pool.map(get_stock_data, database.stocks)

# pool = Pool(cpu_count())
# for stock in database.stocks:
#     pool.apply_async(get_stock_data, args=(stock), callback=store_data)
# pool.close()
# pool.join()
