from financial_calcs import round_date
from datetime import date, datetime, timedelta
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import math


class Stock_Screener:
    def __init__(self, sp500_stocks):
        self.sp500_stocks = sp500_stocks

        # Index Returns
        start_date = datetime.now() - timedelta(days=365)
        end_date = date.today()

        index_name = '^GSPC'
        index_df = pdr.get_data_yahoo(index_name, start_date, end_date)
        index_df['Percent Change'] = index_df['prices'].pct_change()
        index_return = (index_df['Percent Change'] + 1).cumprod()[-1]

        returns_multiple = []
        self.returns_multiple_dict = {}
        for name, stock in self.sp500_stocks.items():
            stock_return = list((stock.percent_change + 1).cumprod())[-1]
            returns_multiple.append(round((stock_return / index_return), 2))
            self.returns_multiple_dict[name] = round((stock_return / index_return), 2)

        # self.rank_returns = {y[0]:x for x, y in sorted(enumerate(returns_multiple), key=lambda x: x[1][1])}
        # self.returns = {name:val for name,val in returns_multiple}
        self.top_percentile = np.percentile(np.array(returns_multiple), 70)

        self.filtered_stocks = []
        self.EMA_screener()


    # returns list of all stocks that pass the EMA screener requirements (only returns current stocks, not historical)
    def EMA_screener(self):
        count = 0
        for name, stock in self.sp500_stocks.items():
            # if name != 'UAA':
            #     continue

            cur_date = stock.dates[-1]
            cur_price = stock.prices[-1]

            SMA_50 = stock.SMAs[50]
            SMA_150 = stock.SMAs[150]
            SMA_200 = stock.SMAs[200]
            week_52_low = self.get_week_low(stock, cur_date, weeks=52)
            week_52_high = self.get_week_high(stock, cur_date, weeks=52)

            cond1 = cur_price > SMA_150[-1] and cur_price > SMA_200[-1]
            cond2 = SMA_150[-1] > SMA_200[-1]
            cond3 = all([cur_val > prev_val for prev_val, cur_val in zip(SMA_200[-30:], SMA_200[-29:])])
            cond4 = SMA_50[-1] > SMA_150[-1] and SMA_50[-1] > SMA_200[-1]
            cond5 = cur_price > SMA_50[-1]
            cond6 = cur_price >= week_52_low*1.3
            cond7 = cur_price >= week_52_high*0.75
            # print(self.returns_multiple_dict[name], self.top_percentile)

            cond8 = self.returns_multiple_dict[name] >= self.top_percentile

            # print(f'week low {week_52_low}, week high {week_52_high}')
            # print(cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8)
            if cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8:
                count += 1
                self.filtered_stocks.append(stock)

        print(f'count = {count}')

    def get_week_low(self, stock, end_date, weeks):
        # not possible that start_date is on a weekend since end_date is on a weekday
        start_date = end_date - timedelta(weeks=weeks)
        start_date = round_date(start_date)
        print(start_date, end_date)
        assert(start_date in stock.date_to_index)

        start_index = stock.date_to_index[start_date]
        end_index = stock.date_to_index[end_date]

        return min(stock.prices[start_index: end_index+1])


    def get_week_high(self, stock, end_date, weeks):
        # not possible that start_date is on a weekend since end_date is on a weekday
        start_date = end_date - timedelta(weeks=weeks)
        start_date = round_date(start_date)
        if end_date == datetime(2021, 5, 21).date():
            print(start_date, end_date)
        assert(start_date in stock.date_to_index)

        start_index = stock.date_to_index[start_date]
        end_index = stock.date_to_index[end_date]

        return max(stock.prices[start_index: end_index+1])


    def print(self):
        for stock in self.filtered_stocks:
            print(stock.ticker)

