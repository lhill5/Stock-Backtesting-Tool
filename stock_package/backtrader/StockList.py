import time

from Stock import Stock
from trading_calendar import Calendar
from Stock_Prices import StockPrices
from Stock_Fundamentals import StockFundamentals
from global_decorators import *
import datetime
import pandas as pd


class StockList:
    def __init__(self, SQL, logger, batch_size=1, start_date=None, end_date=None):
        if start_date is None:
            start_date = datetime.date.today() - datetime.timedelta(days=10*365)
        if end_date is None:
            end_date = datetime.date.today()

        self.SQL = SQL
        self.logger = logger
        self.batch_size = batch_size
        self.start_date = start_date
        self.end_date = end_date

        self.market_calendar = Calendar()
        self.today = datetime.date.today()

        self.stocks = {}
        self.tickers = []

        # needed for updating stock data
        self.prices_API = StockPrices(self.SQL)
        self.fundamental_API = StockFundamentals(self.SQL)

        stock_info = self.read_ticker_list(sp500=True)
        self.all_tickers, self.sp500_date_added, self.cik_list = stock_info.ticker, stock_info.date_added, stock_info.cik

        self.cik_dict = {stock: cik for stock, cik in zip(self.tickers, self.cik_list)}
        self.update_stocks = self.tickers

        # load initial batch of stocks (speeds up program instead of loading the entire stock list initially
        self.load_batch()


    def load_batch(self):
        for i in range(self.batch_size):
            stock = self._generate_stocks()
            stock = next(stock)
            self.append(stock)


    def _generate_stocks(self):
        for ticker in self.all_tickers:
            stock = Stock(ticker, self.SQL, self.logger, self.market_calendar, request_start_date=self.start_date, request_end_date=self.end_date)
            yield stock

    def append(self, stock):
        if stock is None:
            return
        self.tickers.append(stock.ticker)
        self.stocks[stock.ticker] = stock

    def search(self, ticker):
        if ticker in self.stocks:
            return self.stocks[ticker]
        else:
            return None

    def update(self):
        self.update_prices()
        self.update_fundamentals()

    def update_prices(self):
        # self.update_stocks = self.update_stocks[:20]
        # ______________________________________________________________________________________________________________
        # copy new data to SQL, if a table doesn't exist for a stock, create it

        stock_data_queue = []

        prev_stock_API_time = time.time()
        stock_API_limit_reached = False

        update_list_len = len(self.update_stocks)
        si = 0  # stock index and fundamental index for keeping track of current ticker

        while (si < update_list_len):
            # get stock prices data
            if si < update_list_len:
                ticker = self.update_stocks[si]
                cik = self.cik_list[si]
                db_ticker = self.get_db_ticker_name(ticker)

                assert (len(self.cik_list == len(self.update_stocks)))

                # check to see if data already exists before API call and SQL table edit
                prices_data = self.SQL.query_table(db_ticker)
                if prices_data and prices_data[-1][0] == self.SQL.latest_trading_date():
                    si += 1
                    continue

                # check if API limit reached
                if si % 300 == 0 and si != 0:
                    stock_API_limit_reached = True

                if stock_API_limit_reached:
                    if (time.time() - prev_stock_API_time >= 60):
                        prev_stock_API_time = time.time()
                        stock_API_limit_reached = False
                    else:
                        if stock_data_queue:
                            queue_data = stock_data_queue.pop()
                            db_ticker, data = list(queue_data.items())[0]
                            if data:
                                self.SQL.update_prices_DB_table(db_ticker, data)
                            else:
                                print(f'{db_ticker} no data')

                if not stock_API_limit_reached:
                    self.prices_API.get_data(ticker)
                    stock_data_queue.append({db_ticker: self.prices_API.get_API_data(ticker)})
                    si += 1

        # update remaining stocks in queue if any left
        while len(stock_data_queue) != 0:
            queue_data = stock_data_queue.pop()
            db_ticker, data = list(queue_data.items())[0]
            if data:
                self.SQL.update_prices_DB_table(db_ticker, data)
            else:
                print(f'{db_ticker} no data')

    def update_fundamentals(self):
        # self.update_stocks = self.update_stocks[:20]
        # ______________________________________________________________________________________________________________

        fundamental_data_queue = []

        prev_stock_API_time = time.time()
        prev_fund_API_time = time.time()

        stock_API_limit_reached = False
        fund_API_limit_reached = False

        update_list_len = len(self.update_stocks)
        fi = 0  # stock index and fundamental index for keeping track of current ticker

        while (fi < update_list_len):

            if fi < update_list_len:
                ticker = self.update_stocks[fi]
                db_ticker = self.get_db_ticker_name(ticker).upper()

                # if SQL table already has data for this ticker, then no need to use API call
                fundamental_data = self.SQL.query_table(db_ticker, fundamental_db=True)
                if fundamental_data:
                    fi += 1
                    continue

                # check to see if API limit has been reached
                if fi % 5 == 0 and fi != 0:
                    if not fund_API_limit_reached:
                        print(f'{fi}: {ticker}')
                        print('API limit reached')
                    fund_API_limit_reached = True

                if fund_API_limit_reached:
                    if (time.time() - prev_fund_API_time >= 60):
                        prev_fund_API_time = time.time()
                        fund_API_limit_reached = False
                    else:
                        if fundamental_data_queue:
                            queue_data = fundamental_data_queue.pop()
                            db_ticker, data = list(queue_data.items())[0]
                            if data:
                                self.SQL.update_fundamental_DB_table(db_ticker, data)
                            else:
                                print(f'{db_ticker} no data')

                if not fund_API_limit_reached:
                    fundamental_data_queue.append({db_ticker: self.fundamental_API.get_API_data(ticker)})
                    fi += 1

        # update remaining stocks in queue if any left
        while len(fundamental_data_queue) != 0:
            queue_data = fundamental_data_queue.pop()
            db_ticker, data = list(queue_data.items())[0]
            if data:
                self.SQL.update_fundamental_DB_table(db_ticker, data)
            else:
                print(f'{db_ticker} no data')

    def read_ticker_list(self, stocks_only=False, US_only=False, sp500=False):
        if US_only:
            query = "SELECT stock FROM stock_list where exchange like '%NYSE%' or exchange like '%NASDAQ%'"
        elif stocks_only:
            query = "SELECT stock FROM stock_only_list"
        elif sp500:
            query = "SELECT stock, date_added, cik FROM sp500_list"
        else:
            query = "SELECT stock FROM stock_list"

        if sp500:
            stock_data = self.SQL.read_query(query, sp500_list=True)
            df = pd.DataFrame(stock_data, columns=['ticker', 'date_added', 'cik'])
            return df

        else:
            stocks = self.SQL.read_query(query)
            df = pd.DataFrame(stocks, columns=['ticker'])
            return df


    def update_stock_list(self):
        # self.delete_table('stock_list')
        # self.create_stock_list_table()
        self.SQL.delete_table('stock_only_list')
        self.SQL.create_stock_only_list_table()

    def update_sp500_stock_list(self):
        self.SQL.delete_table('sp500_list', sp500_list=True)
        self.SQL.create_sp500_list_table()

    def get_sp500_list(self):
        query = "SELECT stock, date_added, cik FROM sp500_list"

        stock_data = self.SQL.read_query(query, sp500_list=True)
        stocks = []
        dates_added = []
        ciks = []
        for stock in stock_data:
            # breakpoint()
            ticker, date_added, cik = stock
            stocks.append(ticker)
            dates_added.append(date_added)
            ciks.append(cik)
        return (stocks, dates_added, ciks)

    def get_russel_2000_list(self):
        pass

    def get_full_stock_list(self):
        pass

    def get_US_stockList(self):
        pass

    def get_International_List(self):
        pass

    def get_db_ticker_name(self, ticker):
        return ticker.replace('-', '').replace('.', '')
    
