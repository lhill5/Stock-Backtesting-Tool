from financial_calcs import get_last_trading_day
import yfinance as yf
import mysql.connector
from mysql.connector import Error
from os.path import exists
import math

import pandas as pd
import datetime
from multiprocessing import Pool, cpu_count
import alpaca_trade_api as tradeapi
import config


class SQL_DB:
    connection = None

    def __init__(self, update=False, update_list=None):
        self.pw = 'mssqlserver44'
        self.db = 'stock_data'
        self.update = update
        self.update_list = update_list
        self.update_all_tickers_list = False

        if not SQL_DB.connection:
            # self.create_database(f"CREATE DATABASE {self.db}", self.pw)
            SQL_DB.connection = self.connect_to_database("localhost", "root", self.pw, self.db)

        if self.update_all_tickers_list:
            self.delete_table('stock_list')
            self.create_stock_list_table()

        query = "SELECT stock FROM stock_list"
        self.stocks = self.read_query(query)
        self.stocks = [stock[0] for stock in self.stocks]
        print(f'{len(self.stocks)} stocks in sql db')

        if self.update or self.update_list is not None:
            self.debug = False
            if self.debug:
                self.stocks = ['FVIV', 'DTM-WI', 'CHAQ-U', 'BST-RT']
            elif self.update_list is not None:
                self.stocks = self.update_list

            self.latest_trading_date = get_last_trading_day() #- datetime.timedelta(days=1)
            self.latest_stock_date = {stock: self.get_latest_sql_date(stock) for stock in self.stocks}
            self.earliest_stock_date = {stock: self.get_earliest_sql_date(stock) for stock in self.stocks}

            ipo_stocks_in_last_year = [stock for stock in self.stocks if self.earliest_stock_date[stock] is not None and self.earliest_stock_date[stock] > (datetime.date.today() - datetime.timedelta(days = 365*5))]
            self.update_stocks = [stock for stock in self.stocks if self.latest_stock_date[stock] is None or self.latest_stock_date[stock] < self.latest_trading_date]
            self.new_stock_data = {}

            #self.get_stock_data('NDP')
            # self.update_stocks = self.update_stocks[:1]
            print(f'latest trading date: {self.latest_trading_date}')
            print(f'{len(self.stocks)} tables, update {len(self.update_stocks)} tables, {len(ipo_stocks_in_last_year)} IPO stocks in last year')
            #print(ipo_stocks_in_last_year)
            self.sql_update_tables()


    def get_SQL_connection(self):
        return SQL_DB.connection


    # creates a fact table of all stocks currently being used
    def create_stock_list_table(self):
        api = tradeapi.REST(
            config.API_KEY,
            config.SECRET_KEY,
            config.PAPER_URL, api_version='v2'
        )

        active_assets = api.list_assets(status='active')
        stocks = [a.symbol.replace('.', '-') for a in active_assets]

        create_table_query = f"""
        CREATE TABLE stock_list (
            stock_index INT PRIMARY KEY,
            stock VARCHAR (8) NOT NULL
        );
        """
        if not self.table_exists('stock_list'):
            self.execute_query(create_table_query)

        insert_data_query = f"""
        INSERT INTO stock_list
        (stock_index, stock)
        VALUES (%s, %s)
        """
        sql_format_stocks = [[i, s] for i, s in enumerate(stocks)]
        self.multiline_query(insert_data_query, sql_format_stocks)


    # creates a table in the 'stock_data' database for the ticker parameter
    def sql_table_push(self, ticker):
        ticker = ticker.upper()
        ticker_letters = ticker.replace('-', '') # mysql table names cannot contain special character '-'

        create_table_query = f"""
            CREATE TABLE {ticker_letters}_table (
                date DATE PRIMARY KEY,
                open DECIMAL(10,2),
                high DECIMAL(10,2),
                low DECIMAL(10,2),
                close DECIMAL(10,2),
                adj_close DECIMAL(10,2),
                volume BIGINT
            );
            """

        # create table in database if it doesn't already exist
        if not self.table_exists(f'{ticker_letters}_table'):
            self.execute_query(create_table_query)
            print(f'created {ticker_letters}_table')

        insert_data_query = f"""
            INSERT INTO {ticker_letters}_table
            (date, open, high, low, close, adj_close, volume) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

        file_exists = exists(f'stock_data/{ticker}.csv')
        # if not file_exists:
        #     webscrape_ticker(ticker)
        #     file_exists = exists(f'stock_data/{ticker}.csv')

        if not file_exists:
            print(f'{ticker} csv file not found')
        else:
            df = pd.read_csv(f'stock_data/{ticker}.csv')
            df = df.dropna() # deletes rows with nan in them
            column_values = df.values.tolist()
            self.multiline_query(insert_data_query, column_values)


    def sql_update_tables(self):
        # self.update_stocks = self.update_stocks[:20]
        # create pool of processes to read stock data
        if self.debug:
            pool = None
        else:
            pool = Pool(cpu_count())
        for ticker in self.update_stocks:
            if self.debug:
                ticker, data = self.get_stock_data(ticker)
                self.new_stock_data[ticker] = data
            else:
                pool.apply_async(self.get_stock_data, args=(ticker,), callback=self.log_result)

        if not self.debug:
            pool.close()
            pool.join()

        # reconnect to database (multiprocess disconnects sql connection for some reason)
        SQL_DB.connection = self.connect_to_database("localhost", "root", self.pw, self.db)

        # for stocks not in mysql db, create table and copy csv data into db
        for ticker in self.update_stocks:
            ticker_letters = ticker.replace('-', '')
            # if table doesn't exist or it's a null table, copy csv data to stock db
            if self.latest_stock_date[ticker] is None:
                # self.delete_table(f'{ticker_letters}_table')
                self.sql_table_push(ticker)

            insert_data_query = f"""
                INSERT INTO {ticker_letters}_table
                (date, open, high, low, close, adj_close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            if ticker in self.new_stock_data:
                data = self.new_stock_data[ticker]
                if data is not None and len(data) > 0:
                    self.multiline_query(insert_data_query, data)
            else:
                print(f'{ticker} not in read data')


    def get_stock_data(self, ticker, similar_ticker=None):
        if self.latest_stock_date[ticker] is None:
            return None, None

        print(ticker)
        # get data from yfinance API
        tickerData = yf.Ticker(ticker) if not similar_ticker else yf.Ticker(similar_ticker)
        # get the historical prices for this ticker
        start_date = self.latest_stock_date[ticker] + datetime.timedelta(days=1)
        end_date = self.latest_trading_date + datetime.timedelta(days=1)
        tickerDf = tickerData.history(period='1d', start=str(start_date), end=str(end_date), auto_adjust=False)

        # transform data into list of tuples, where each tuple represents a single day's amount of data (date, open, high, low, close, adj_close, volume)
        dates = [d.date() for d in tickerDf['Open'].keys()]
        open_prices, high_prices, low_prices, close_price, adj_close, volume = [list(tickerDf[k]) for k in tickerDf.keys()[:6]]
        row_data = list(zip(dates, open_prices, high_prices, low_prices, close_price, adj_close, volume))
        # checks to see if duplicate dates are in data and removes them
        row_data = list(filter(lambda row: row[0] >= start_date, row_data))

        if len(tickerDf['Open']) == 0 and len(ticker) > 2 and ticker[-2:] == '-U' and similar_ticker is None:
            return self.get_stock_data(ticker, ticker + 'N')
        else:
            return (ticker, row_data)

        # response = requests.get(url)
        # html_page = response.content
        # soup = BeautifulSoup(html_page, 'html.parser')
        #
        # # see if symbol couldn't be found and a similar symbol was recommended by yahoo-finance
        # table_val = None
        # try:
        #     table_val = list(soup.find_all('tr')[1])[0].text
        # except IndexError as e:
        #     if len(soup.find_all('tr')) != 0 and 'Will be right back' in list(soup.find_all('tr'))[0].text:
        #         print(list(soup.find_all('tr'))[0].text)
        #         print('yahoo-finance website is down, stopping ...')
        #         raise ValueError(e)
        #     else:
        #         print(f'{ticker} cannot be found in yahoo-finance')

        # try:
        #     # attempts to convert first row of table to a date, if this fails then ticker wasn't found
        #     date = datetime.datetime.strptime(table_val.replace(',', ''), "%b %d %Y").date()
        #
        #     ticker_data = []
        #     latest_sql_date = self.latest_stock_date[ticker]
        #     for row in soup.find_all('tr'):
        #         try:
        #             date, open_price, high_price, low_price, close_price, adj_close, volume = [r.text.replace(',', '') for r in row.children]
        #             date = datetime.datetime.strptime(date, "%b %d %Y").date()
        #             if date <= latest_sql_date:
        #                 break
        #
        #             volume = volume.replace(',', '')
        #             volume = 0 if volume == '-' else int(volume)
        #             open_price, high_price, low_price, close_price, adj_close = float(open_price), float(high_price), float(low_price), float(close_price), float(adj_close)
        #
        #             ticker_data.append((date, open_price, high_price, low_price, close_price, adj_close, volume))
        #         except:
        #             pass
        #     return (ticker, ticker_data)
        #
        # except:
        #     # if fails to convert table_val to a date, then ticker wasn't found
        #     # similar ticker is None prevents recursive infinite loop, since if similar ticker wasn't found, no need to continue searching for similar ticker
        #     if similar_ticker is None and ticker in table_val:
        #         return self.get_recent_data(ticker, table_val)
        #     else:
        #         return (ticker, None)


    def log_result(self, result):
        ticker, data = result
        self.new_stock_data[ticker] = data


    def connect_to_database(self, host_name, user_name, user_password, db_name=None):
        connection = None
        try:
            if db_name:
                connection = mysql.connector.connect(
                    host=host_name,
                    user=user_name,
                    passwd=user_password,
                    database=db_name
                )
            else:
                connection = mysql.connector.connect(
                    host=host_name,
                    user=user_name,
                    passwd=user_password,
                )
            print("MySQL Database SQL_DB.connection successful")

        except Error as err:
            print(f"Error: '{err}'")

        return connection


    def create_database(self, query, pw):
        connection = self.connect_to_database("localhost", "root", pw)
        cursor = connection.cursor()

        try:
            cursor.execute(query)
            print("Database created successfully")
        except Error as err:
            # print(f"Error: '{err}'")
            return -1

        return 0


    def execute_query(self, query):
        cursor = SQL_DB.connection.cursor()
        try:
            cursor.execute(query)
            SQL_DB.connection.commit()  # write to db
            # print("Query successful")
        except Error as err:
            print(f"Error: '{err}'")
            return -1

        return 0


    def multicol_query(self, query, vals):
        cursor = SQL_DB.connection.cursor()
        try:
            cursor.execute(query, vals)
            SQL_DB.connection.commit()  # write to db
            # print("Query successful")
        except Error as err:
            print(f"Error: '{err}'")
            return -1

        return 0


    def multiline_query(self, query, column_values):
        cursor = SQL_DB.connection.cursor()
        try:
            cursor.executemany(query, column_values)
            SQL_DB.connection.commit()  # write to db
            # print("Query successful")
        except Error as err:
            print(f"Error: '{err}'")
            return -1

        return 0


    def read_query(self, query):
        cursor = SQL_DB.connection.cursor()
        result = None
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            return result
        except Error as err:
            print(f"Error: '{err}'")
            return -1


    def table_exists(self, table_name):
        query = f"SELECT * FROM {table_name}"
        cursor = SQL_DB.connection.cursor(buffered=True)
        try:
            cursor.execute(query)
            return 1
        except Error as err:
            return 0


    def column_exists(self, ticker, col):
        ticker = ticker.replace('-', '')

        query = f"show columns from stock_data.{ticker}_table like '{col}'"
        rst = self.read_query(query)
        return rst != [] and rst != -1


    def get_latest_sql_date(self, ticker):
        ticker = ticker.replace('-', '')

        query = f"""
        SELECT date
        FROM {ticker}_table
        order by date desc
        limit 1
        """
        latest_row = self.read_query(query)
        if latest_row == -1 or latest_row == []:
            return None
        return latest_row[0][0]


    def get_earliest_sql_date(self, ticker):
        ticker = ticker.replace('-', '')

        query = f"""
        SELECT date
        FROM {ticker}_table
        order by date asc
        limit 1
        """
        latest_row = self.read_query(query)
        if latest_row == -1 or latest_row == []:
            return None
        return latest_row[0][0]


    def get_latest_non_null_col(self, ticker, col):
        ticker = ticker.replace('-', '')

        query = f"""
        SELECT date
        FROM {ticker}_table
        WHERE {col} is not null
        ORDER BY date desc
        LIMIT 1
        """
        latest_row = self.read_query(query)
        if latest_row == -1 or latest_row == []:
            return None
        return latest_row[0][0]


    def get_new_col_query(self, ticker, col):
        ticker = ticker.replace('-', '')

        insert_col_query = f"""
        ALTER TABLE stock_data.{ticker}_TABLE
        ADD {col} VARCHAR(10);
        """
        return insert_col_query


    def get_insert_data_query(self, ticker, col):
        ticker = ticker.replace('-', '')

        insert_data_query = f"""
        INSERT INTO {ticker}_table
        ({col})
        VALUES (%s)
        """
        return insert_data_query


    def sql_format_data(self, data):
        formatted_data = [[None] if math.isnan(d) else [d] for d in data]
        return formatted_data


    def update_row(self, ticker, row_id, cols, vals):
        ticker_letters = ticker.replace('-', '')

        vals_list = list(vals)
        for i, val in enumerate(vals_list):
            vals_list[i] = val if val is None or isinstance(val, datetime.date) else round(val, 5)
        vals = tuple(vals_list)

        sel_cols = ""
        for i, col in enumerate(cols):
            sel_cols += f'{col}=%s'
            if col != cols[-1]:
                sel_cols += ', '

        query = f"""
        UPDATE {ticker_letters}_table
        SET {sel_cols}
        WHERE date=%s
        """
        self.multicol_query(query, vals)


    def delete_table(self, table_name):
        query = f"""
            DROP TABLE {table_name}
        """
        self.execute_query(query)

