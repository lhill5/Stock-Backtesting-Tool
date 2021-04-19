import yfinance as yf
from datetime import datetime
from yahoo_earnings_calendar import YahooEarningsCalendar
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr


def avg(list):
    return sum(list) / len(list)


def stock_lookup(stock):
    ticker = yf.Ticker(stock)

    history = ticker.history(period="max")
    dates, prices = [], []
    # get data
    for key, value in history.items():
        if key == "Close":
            dd = defaultdict(list)
            arr = value.to_dict(dd)
            for timestamp, price in arr.items():
                date = timestamp.date()
                dates.append(date)
                prices.append(price)

    return dates, prices


# yec = YahooEarningsCalendar(0)
# next_earnings_date = datetime.utcfromtimestamp(yec.get_next_earnings_date('aapl')).strftime('%Y-%m-%d %H:%M:%S')
#
# earnings = yec.get_earnings_of('aapl')
# quarterly_dates = []
# quarterly_EPS = []
# for data in earnings:
#     date = datetime.strptime(data['startdatetime'][:10], '%Y-%m-%d')
#     actual_eps = data['epsactual']
#     if actual_eps is not None:
#         quarterly_dates.append(date)
#         quarterly_EPS.append(actual_eps)
#
#
# quarterly_dates.reverse()
# for date in quarterly_dates:
#     print(date)
#
# quarterly_EPS.reverse()
# rolling_avg = [avg(quarterly_EPS[i-3:i+1]) if (i-3) >= 0 else np.nan for i in range(len(quarterly_EPS))]
#
# dates, prices = stock_lookup('aapl')
# PE = []
#
# print(quarterly_dates[0], quarterly_EPS[0])
# index = 3
# len_EPS = len(quarterly_dates)
#
# for date, price in zip(dates, prices):
#     if date < quarterly_dates[3].date():
#         PE.append(np.nan)
#
#     else:
#         if (index+1) < len_EPS and date > quarterly_dates[index+1].date():
#             index += 1
#
#         if index + 1 < len_EPS:
#             print(date, quarterly_dates[index].date(), quarterly_dates[index+1].date())
#
#         if date >= quarterly_dates[index].date() and (index+1) < len_EPS and date < quarterly_dates[index+1].date():
#             PE_val = price / quarterly_EPS[index] if quarterly_EPS[index] != 0 else 0
#             PE.append(PE_val)
#
#         else:
#             PE.append(np.nan)


# ax = plt.subplot(111)
# ax2 = ax.twinx()
# ax.plot(dates, prices, color='g')
# ax.plot(dates, PE, color='r')
#
# print(quarterly_EPS)
# print(rolling_avg)
# ax2.plot(quarterly_dates, rolling_avg, color='b')
# plt.show()

stock = 'see'
yf.pdr_override()
data = pdr.get_data_yahoo(stock)

print(data)
