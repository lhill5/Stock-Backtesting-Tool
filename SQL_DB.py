import mysql.connector
from mysql.connector import Error
import pandas as pd
import datetime
from os.path import exists
import math


class SQL_DB:
    connection = None

    def __init__(self, update=False):
        self.pw = 'mssqlserver44'
        self.db = 'stock_data'
        self.update = update

        if not SQL_DB.connection:
            self.create_database(f"CREATE DATABASE {self.db}", self.pw)
            SQL_DB.connection = self.connect_to_database("localhost", "root", self.pw, self.db)

        if not self.table_exists('stock_list'):
            self.create_stock_list_table()

        query = "SELECT stock FROM stock_list"
        self.stocks = self.read_query(query)
        self.stocks = [stock[0] for stock in self.stocks]

        if self.update:
            self.get_stock_list_data()


    def get_SQL_connection(self):
        return SQL_DB.connection


    def get_stock_list_data(self):
        for ticker in self.stocks:
            self.get_ticker_data(ticker)


    def get_ticker_data(self, ticker):
        last_trading_day = get_last_trading_day()

        ticker_letters = ticker.replace('-', '')

        # copies ticker data into the 'stock_data' sql database to be used later
        latest_sql_date = self.get_latest_sql_date(ticker)
        if self.table_exists(ticker_letters + '_table') and latest_sql_date is not None:
            # checks if sql table has the latest stock data or not, if not then update table to latest data
            if last_trading_day != latest_sql_date:
                # update sql table
                pass
        else:
            print(f'created {ticker_letters}_table')
            self.sql_push(ticker)


    # creates a fact table of all stocks currently being used
    def create_stock_list_table(self):
        tickers = pd.read_csv('s&p_500.csv', names=['ticker symbols'])
        stocks = []
        for col, stock_name in tickers.iterrows():
            ticker, name, *_ = stock_name[0].split(' - ')
            ticker = ticker.replace('.', '-')  # brk.b -> brk-b (format yahoo-finance uses)
            stocks.append([ticker])

        create_table_query = f"""
        CREATE TABLE stock_list (
            stock VARCHAR (6) PRIMARY KEY NOT NULL
        );
        """
        self.execute_query(create_table_query)

        insert_data_query = f"""
        INSERT INTO stock_list
        (stock) 
        VALUES (%s)
        """
        self.multiline_query(insert_data_query, stocks)


    # creates a table in the 'stock_data' database for the ticker parameter
    def sql_push(self, ticker):
        ticker = ticker.upper()
        ticker_letters = ticker.replace('-', '') # mysql table names cannot contain special character '-'

        create_table_query = f"""
            CREATE TABLE {ticker_letters}_table (
                date DATE PRIMARY KEY,
                open DECIMAL(7,2),
                high DECIMAL(7,2),
                low DECIMAL(7,2),
                close DECIMAL(7,2),
                adj_close DECIMAL(7,2),
                volume BIGINT
            );
            """
        # create table in database, if it already exists then return (no-data to write to sql db)
        self.execute_query(create_table_query)

        insert_data_query = f"""
            INSERT INTO {ticker_letters}_table
            (date, open, high, low, close, adj_close, volume) 
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

        file_exists = exists(f'stock_data/{ticker}.csv')
        if file_exists:
            df = pd.read_csv(f'stock_data/{ticker}.csv')
            df = df.dropna() # deletes rows with nan in them
            column_values = df.values.tolist()
            self.multiline_query(insert_data_query, column_values)


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


def get_last_trading_day():
    today = datetime.date.today()
    weekday = today.weekday()
    # if today is a weekend
    if weekday > 4:
        rst = today - datetime.timedelta(days=weekday-4)
    else:
        rst = today
    return rst

