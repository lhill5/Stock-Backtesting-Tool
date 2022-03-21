from collections import defaultdict
from mysql.connector import Error
import mysql.connector
import random


class SQL_DB:
    stock_data_connection = None
    fundamental_data_connection = None
    sp500_list_connection = None

    def __init__(self, stock_data_API, update=False, update_list=None):

        # update_list = ['MIK']
        self.stock_data_API = stock_data_API

        self.host_name = "localhost"
        self.user_name = "root"
        self.user_password = 'mssqlserver44'
        self.stock_db = 'stock_data'
        self.fundamental_db = 'fundamental_data'
        self.sp500_db = 'stock_list'

        self.update = update
        self.update_list = update_list
        self.debug = False
        self.debug2 = False
        self.daily_API_limit_reached = False

        self.dates = []
        self.prices = defaultdict(list)

        if not SQL_DB.stock_data_connection:
            SQL_DB.stock_data_connection = self.connect_to_database(self.stock_db)
        if not SQL_DB.fundamental_data_connection:
            SQL_DB.fundamental_data_connection = self.connect_to_database(self.fundamental_db)
        if not SQL_DB.sp500_list_connection:
            SQL_DB.sp500_list_connection = self.connect_to_database(self.sp500_db)

        # breakpoint()
        # self.update_stock_list()
        # self.update_sp500_stock_list()

        # self.stocks = self.stock_data_API.get_stock_list()
        # self.stocks = self.read_stock_list()[14700:14720]
        if self.update_list is not None:
            self.stocks = self.update_list
        else:
            # stocks, cik
            self.stocks, self.sp500_date_added, self.cik_list = self.read_stock_list(sp500=True)
            self.cik_dict = {stock: cik for stock, cik in zip(self.stocks, self.cik_list)}

            # self.stocks, self.sp500_date_added, self.cik_list = self.find_sp500_stock('JPM')


    def connect_to_database(self, db_name=None):
        connection = None
        try:
            if db_name:
                connection = mysql.connector.connect(
                    host=self.host_name,
                    user=self.user_name,
                    passwd=self.user_password,
                    database=db_name,
                    connect_timeout=1000,
                )
            else:
                connection = mysql.connector.connect(
                    host=self.host_name,
                    user=self.user_name,
                    passwd=self.user_password,
                )
            print("MySQL Database SQL_DB.connection successful")

        except Error as err:
            print(f"Error: '{err}'")
            if self.debug:
                breakpoint()

        return connection


    def read_query(self, query, stock_db=True, fundamental_db=False, sp500_list=False):
        if sp500_list:
            db = SQL_DB.sp500_list_connection
        elif fundamental_db:
            db = SQL_DB.fundamental_data_connection
        else:
            db = SQL_DB.stock_data_connection

        cursor = db.cursor()
        try:
            # df = cx.read_sql(f"mysql://{self.user_name}:{self.user_password}@localhost:3306/{self.db}", query)
            # df = df.values.tolist()
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Error as err:
            print(f"Error: '{err}'")
            if self.debug:
                breakpoint()
            return -1


    def read_stock_list(self, stocks_only=False, US_only=False, sp500=False):
        if US_only:
            query = "SELECT stock FROM stock_list where exchange like '%NYSE%' or exchange like '%NASDAQ%'"
        elif stocks_only:
            query = "SELECT stock FROM stock_only_list"
        elif sp500:
            query = "SELECT stock, date_added, cik FROM sp500_list"
        else:
            query = "SELECT stock FROM stock_list"

        if sp500:
            stock_data = self.read_query(query, sp500_list=True)
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

        else:
            stocks = self.read_query(query)
            stocks = [stock[0] for stock in stocks]
            return stocks


    def query_stock_prices(self, ticker):
        ticker = ticker.replace('-', '').replace('.', '')

        cols = "date, open, high, low, close, adj_close, volume"
        query = f"SELECT {cols} FROM {ticker}_table"
        # query = f"SELECT {cols} FROM {self.ticker_letters}_table limit 1"

        query_output = self.read_query(query)
        if query_output != -1 and query_output != []:
            for row in query_output:
                date, open_price, high_price, low_price, close_price, adj_close_price, volume = row
                self.dates.append(date)
                self.prices['open'].append(float(open_price))
                self.prices['high'].append(float(high_price))
                self.prices['low'].append(float(low_price))
                self.prices['close'].append(float(close_price))
                self.prices['adj_close'].append(float(adj_close_price))
                self.prices['volume'].append(volume)
        else:
            print(f'error reading from {ticker}_table')

        if len(self.prices['open']) == 0:
            return -1


def compare_data(db_dates, db_prices, start_date, end_date):
    oprices, hprices, lprices, cprices, vol = db_prices['open'], db_prices['high'], db_prices['low'], db_prices['close'], db_prices['volume']


def get_alpha_vantage_data():
    pass

def get_iex_data():
    pass

def get_twelve_data():
    pass

def get_fmp_data():
    pass

def get_tiingo_data():
    pass

def get_EOD_data():
    pass


if __name__ == "__main__":
    db = SQL_DB(None)
    ticker = random.choice(db.stocks)

    # query data
    db.query_stock_prices(ticker)
    db_dates, db_prices = db.dates, db.prices

    start_date, end_date = db_dates[0], db_dates[-1]

    compare_data(db_dates, db_prices, start_date, end_date)