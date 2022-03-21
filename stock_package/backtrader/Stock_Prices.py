import json
import time
import requests

import pandas as pd
from alpha_vantage.timeseries import TimeSeries

from global_functions import get_db_ticker
from global_decorators import *


class StockPrices:
    def __init__(self, database):
        self.API_KEY = 'e0ecb6fd8cf50faf985ffe9602bb8fc1'
        self.POLYGON_API_KEY = 'SEp3jMzPKmbxBD_UuJw2uYxjfWgqkoaT'
        self.base_url = 'https://financialmodelingprep.com'
        self.ALPHA_VANTAGE_API_KEY = 'SSFPLDGNSX4NAWV8'
        self.SQL = database

        # each instance of prices API will correspond with a specific ticker's prices
        # self.dates = self.data.date
        # self.open = self.data.open
        # self.high = self.data.high
        # self.low = self.data.low
        # self.close = self.data.close
        # self.adj_close = self.data.adj_close
        # self.volume = self.data.volume
        # self.ohlcv = {'o': self.open, 'h': self.high, 'l': self.low, 'c': self.close, 'a': self.adj_close, 'v': self.volume}


    def get_data(self, ticker, start_date=None, end_date=None):
        db_ticker = get_db_ticker(ticker)

        if self.SQL.table_exists(ticker=db_ticker):
            data = self.get_SQL_data(ticker, start_date=start_date, end_date=end_date)
        else:
            data = self._request_API_data(ticker)

        return data


    def get_SQL_data(self, ticker, start_date=None, end_date=None):
        db_ticker = get_db_ticker(ticker)
        data = self.SQL.query_prices(db_ticker, start_date=start_date, end_date=end_date)
        return data


    def get_API_data(self, ticker):
        return self._request_API_data(ticker)


    def _request_API_data(self, ticker):
        return None
        # return self.get_adjusted_daily(ticker)


    def get_adjusted_daily(self, ticker):
        ts = TimeSeries(key=self.ALPHA_VANTAGE_API_KEY,  # claim on alphavantage.co
                        output_format='pandas',
                        indexing_type='date')

        daily = ts.get_daily_adjusted(ticker, 'full')[0].sort_index()
        daily.columns = [x[3:] for x in daily.columns]
        return ts


    def _calculate_adjusted(self, df, dividends=False):
        # source_url = 'https://www.tradewithscience.com/stock-split-adjusting-with-python/'

        # we will go from today to the past
        new = df.sort_index(ascending=False)

        split_coef = new['split coefficient'].shift(1
                                                    ).fillna(1).cumprod()

        for col in ['open', 'high', 'low', 'close']:
            new['adj_' + col] = new[col] / split_coef
        new['adj_volume'] = split_coef * new['volume']

        if dividends:
            new['adj_dividends'] = new['dividend amount'] / split_coef

        return new.sort_index(ascending=True)


    def print(self, data, all=False):
        pass
        if data is None:
            return

        if not all:
            print(data)
        else:
            # prints all rows/cols
            with pd.option_context('display.max_rows', None,
                                   'display.max_columns', None,
                                   'display.precision', 3):
                print(data)


    def parse_stockList_json(self, json_data):
        stock_list = []
        for stock in json_data:
            exchange = stock['exchange']
            exchange = exchange.replace('New York Stock Exchange', 'NYSE')
            exchange = exchange.replace('Nasdaq', 'NASDAQ')
            stock_list.append((stock['symbol'], exchange))
        return stock_list


    # todo - add to stockList module, figure out which API this uses
    # def get_tradeable_stock_list(self):
    #     endpoint = 'api/v3/available-traded/list'
    #     url = self.base_url + f'/{endpoint}?apikey={self.API_KEY}'
    #     json_data = self.getResponseJson(url)
    #
    #     stocks = self.parse_stockList_json(json_data)
    #     return stocks


    def get_stock_list(self):
        # endpoint = 'api/v3/available-traded/list'
        endpoint = 'api/v3/stock/list'
        url = self.base_url + f'/{endpoint}?apikey={self.API_KEY}'
        r = requests.get(url)
        json_data = r.json()

        # json_data = self.getResponseJson(url)
        stocks = self.parse_stockList_json(json_data)
        with open('stock_list.txt', 'w') as f:
            for stock in stocks:
                f.write(stock)
                f.write('\n')
        return stocks


    def get_sp500_list(self):
        sp500_df = pd.read_csv('../Stock_Data_API/s&p_500.csv', header=None, names=['ticker_name'])
        sp500_df[['ticker', 'name', 'date added', 'cik']] = sp500_df.ticker_name.str.split(" - ", expand=True)
        sp500_df.drop('ticker_name', axis=1, inplace=True)

        # in case stock was added multiple times to the sp500, get most recent date (1st date)
        sp500_df['date added'] = [d.split(' ')[0] for d in sp500_df['date added']]
        return sp500_df.values.tolist()


    def get_stock_only_list(self):
        url = f'https://api.polygon.io/v3/reference/tickers?type=CS&sort=ticker&order=asc&limit=1000&apiKey={self.POLYGON_API_KEY}'
        next_url = url

        stock_list = []
        start_time = time.time()
        counter = 0

        while next_url:
            r = requests.get(next_url)
            data = r.json()
            pretty_data = json.dumps(data, indent=4)
            print(pretty_data)
            counter += 1

            # prevent more than 5 API requests per minute
            if counter % 5 == 0:
                while time.time() - start_time < 60:
                    pass
                start_time = time.time()

            elif data['status'] == 'OK':
                # add stocks to stocks list
                stock_list.extend(data['results'])

                # if more data, get next url and continue adding stocks
                next_url = None
                if 'next_url' in data:
                    next_url = data['next_url']
                    next_url += f'&type=CS&active=true&sort=ticker&order=asc&limit=1000&apiKey={self.POLYGON_API_KEY}'

            elif data['status'] == 'ERROR':
                print('error processing API request (too many requests)')
                time.sleep(5)

        # turn stock list into a list of tuples
        stock_list_tuple = []
        for stock in stock_list:
            stock_list_tuple.append((stock['ticker'], stock['primary_exchange']))
        return stock_list_tuple


    # todo - add to stockList module
    # def get_etf_list(self):
    #     endpoint = 'api/v3/etf/list'
    #     url = self.base_url + f'/{endpoint}?apikey={self.API_KEY}'
    #     json_data = self.getResponseJson(url)
    #     stocks = self.parse_stockList_json(json_data)
    #     with open('etf_list.txt', 'w') as f:
    #         for stock in stocks:
    #             f.write(stock)
    #             f.write('\n')
    #     return stocks


    # @multimethod
    # def get_historical_daily(self, ticker):
    #     endpoint = 'api/v3/historical-price-full'
    #     url = self.base_url + f'/{endpoint}/{ticker}?apikey={self.API_KEY}'
    #     json_data = self.getResponseJson(url)
    #     return self.parse_ohlc_json(json_data)
    #
    #
    # @multimethod
    # def get_historical_daily(self, ticker, start_date: datetime.date, end_date: datetime.date):
    #     start_date, end_date = str(start_date), str(end_date)
    #     endpoint = 'api/v3/historical-price-full'
    #     filter = f'from={start_date}&to={end_date}'
    #     url = self.base_url + f'/{endpoint}/{ticker}?{filter}&apikey={self.API_KEY}'
    #     json_data = self.getResponseJson(url)
    #     return self.parse_ohlc_json(json_data)
    #
    #
    # @multimethod
    # def get_historical_daily(self, ticker, num_rows: int):
    #     endpoint = 'api/v3/historical-price-full'
    #     filter = f'timeseries={num_rows}'
    #     url = self.base_url + f'/{endpoint}/{ticker}?{filter}&apikey={self.API_KEY}'
    #     json_data = self.getResponseJson(url)
    #     return self.parse_ohlc_json(json_data)


if __name__ == '__main__':
    pass
