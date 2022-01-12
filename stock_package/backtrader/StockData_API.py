from multimethod import multimethod
from urllib.error import HTTPError
import urllib.request
import datetime
import json
import requests
import time


class StockData_API:
    def __init__(self):
        self.API_KEY = 'e0ecb6fd8cf50faf985ffe9602bb8fc1'
        self.POLYGON_API_KEY = 'SEp3jMzPKmbxBD_UuJw2uYxjfWgqkoaT'
        self.base_url = 'https://financialmodelingprep.com'

    def getResponseJson(self, url):
        operUrl = urllib.request.urlopen(url)
        jsonData = None
        if(operUrl.getcode()==200):
            data = operUrl.read()
            jsonData = json.loads(data)
        else:
            print("Error receiving data", operUrl.getcode())

        return jsonData


    def parse_ohlc_json(self, json_data):
        if json_data == {}:
            return []

        ticker, data = json_data.values()
        row_data = []
        for row in data:
            row_data.append((row['date'], row['open'], row['high'], row['low'], row['close'], row['adjClose'], row['volume']))
        return row_data


    def parse_stockList_json(self, json_data):
        stock_list = []
        for stock in json_data:
            exchange = stock['exchange']
            exchange = exchange.replace('New York Stock Exchange', 'NYSE')
            exchange = exchange.replace('Nasdaq', 'NASDAQ')
            stock_list.append((stock['symbol'], exchange))
        return stock_list


    def get_tradeable_stock_list(self):
        endpoint = 'api/v3/available-traded/list'
        url = self.base_url + f'/{endpoint}?apikey={self.API_KEY}'
        json_data = self.getResponseJson(url)

        stocks = self.parse_stockList_json(json_data)
        return stocks


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


    def get_etf_list(self):
        endpoint = 'api/v3/etf/list'
        url = self.base_url + f'/{endpoint}?apikey={self.API_KEY}'
        json_data = self.getResponseJson(url)
        stocks = self.parse_stockList_json(json_data)
        with open('etf_list.txt', 'w') as f:
            for stock in stocks:
                f.write(stock + '\n')
        return stocks


    @multimethod
    def get_historical_daily(self, ticker):
        endpoint = 'api/v3/historical-price-full'
        url = self.base_url + f'/{endpoint}/{ticker}?apikey={self.API_KEY}'
        json_data = self.getResponseJson(url)
        return self.parse_ohlc_json(json_data)


    @multimethod
    def get_historical_daily(self, ticker, start_date: datetime.date, end_date: datetime.date):
        start_date, end_date = str(start_date), str(end_date)
        endpoint = 'api/v3/historical-price-full'
        filter = f'from={start_date}&to={end_date}'
        url = self.base_url + f'/{endpoint}/{ticker}?{filter}&apikey={self.API_KEY}'
        json_data = self.getResponseJson(url)
        return self.parse_ohlc_json(json_data)


    @multimethod
    def get_historical_daily(self, ticker, num_rows: int):
        endpoint = 'api/v3/historical-price-full'
        filter = f'timeseries={num_rows}'
        url = self.base_url + f'/{endpoint}/{ticker}?{filter}&apikey={self.API_KEY}'
        json_data = self.getResponseJson(url)
        return self.parse_ohlc_json(json_data)


if __name__ == '__main__':
    stockData_API = StockData_API()
    # stocks = stockData_API.get_tradeable_stock_list()
    # # 0E1L.L is a international stock which isn't supported in free API license
    # print('0E1L.L' in stocks)
    #
    # data = None
    # try:
    #     data = stockData_API.get_historical_daily('EMP', datetime.date(2016, 1, 1), datetime.date(2021, 8, 5))
    # except HTTPError:
    #     print(HTTPError)
    #
    # for val in data:
    #     print(val)

    # stocks = stockData_API.get_tradeable_stock_list()
    # exchanges = set()
    # for stock in stocks:
    #     exchange = stock[1]
    #     exchange = exchange.replace('New York Stock Exchange', 'NYSE')
    #     exchange = exchange.replace('Nasdaq', 'NASDAQ')
    #     exchanges.add(exchange)
    # print(exchanges)
    # print(stocks)

    stockData_API.get_stock_list()
    stockData_API.get_etf_list()

    # start_date, end_date = datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)
    # stockData_API.get_historical_daily(start_date, end_date)
    #
    # stockData_API.get_historical_daily(1)

