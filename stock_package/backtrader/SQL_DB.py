# try:
#     from stock_package.backtrader.financial_calcs import get_last_trading_day
# except ImportError:
#     from financial_calcs import get_last_trading_day

import math
import time
import datetime
import pandas as pd

import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine

from financial_calcs import get_last_trading_day
from global_functions import get_db_ticker
from global_decorators import *


class SQL_DB:
    stock_data_connection = None
    read_only_stock_data_connection = None
    fundamental_data_connection = None
    read_only_fundamental_data_connection = None
    sp500_list_connection = None

    def __init__(self):

        start_time = time.time()

        # update_list = ['MIK']

        self.host_name = "localhost"
        self.user_name = "root"
        self.user_password = 'mssqlserver44'
        self.stock_db = 'stock_data'
        self.fundamental_db = 'fundamental_data'
        self.sp500_db = 'stock_list'
        self.port_number = '3306'

        self.debug = False
        self.debug2 = False
        self.daily_API_limit_reached = False

        if not SQL_DB.stock_data_connection:
            SQL_DB.stock_data_connection = self.connect_to_mysql_database(self.stock_db)
        if not SQL_DB.fundamental_data_connection:
            SQL_DB.fundamental_data_connection = self.connect_to_mysql_database(self.fundamental_db)
        if not SQL_DB.sp500_list_connection:
            SQL_DB.sp500_list_connection = self.connect_to_mysql_database(self.sp500_db)
        if not SQL_DB.read_only_stock_data_connection:
            SQL_DB.read_only_stock_data_connection = self.connect_to_sqlalchemy_database(self.stock_db)
        if not SQL_DB.read_only_fundamental_data_connection:
            SQL_DB.read_only_fundamental_data_connection = self.connect_to_sqlalchemy_database(self.fundamental_db)

        self.latest_trading_date = get_last_trading_day()  # - datetime.timedelta(days=1)


    def get_SQL_connection(self):
        return SQL_DB.stock_data_connection


    # creates a fact table of all stocks currently being used
    # todo - figure out where this funcs below should go
    # def create_stock_list_table(self):
    #
    #     stocks = self.stock_data_API.get_stock_list()
    #
    #     create_table_query = f"""
    #     CREATE TABLE stock_list (
    #         stock_index INT PRIMARY KEY,
    #         stock VARCHAR (16) NOT NULL,
    #         exchange VARCHAR (32) NOT NULL
    #     );
    #     """
    #     if not self.table_exists(table='stock_list'):
    #         self.execute_query(create_table_query)
    #
    #     insert_data_query = f"""
    #     INSERT INTO stock_list
    #     (stock_index, stock, exchange)
    #     VALUES (%s, %s, %s)
    #     """
    #
    #     sql_format_stocks = [[i, s[0], s[1]] for i, s in enumerate(stocks)]
    #     self.multiline_query(insert_data_query, sql_format_stocks)


    # def create_stock_only_list_table(self):
    #
    #     stocks = self.stock_data_API.get_stock_only_list()
    #
    #     create_table_query = f"""
    #     CREATE TABLE stock_only_list (
    #         stock_index INT PRIMARY KEY,
    #         stock VARCHAR (16) NOT NULL,
    #         exchange VARCHAR (32) NOT NULL
    #     );
    #     """
    #     if not self.table_exists(table='stock_only_list'):
    #         self.execute_query(create_table_query)
    #
    #     insert_data_query = f"""
    #     INSERT INTO stock_only_list
    #     (stock_index, stock, exchange)
    #     VALUES (%s, %s, %s)
    #     """
    #
    #     sql_format_stocks = [[i, s[0], s[1]] for i, s in enumerate(stocks)]
    #     self.multiline_query(insert_data_query, sql_format_stocks)


    # def create_sp500_list_table(self):
    #
    #     stocks = self.stock_data_API.get_sp500_list()
    #
    #     create_table_query = f"""
    #     CREATE TABLE sp500_list (
    #         stock_index INT PRIMARY KEY,
    #         stock VARCHAR (16) NOT NULL,
    #         name VARCHAR (64) NOT NULL,
    #         date_added VARCHAR(12) NOT NULL,
    #         CIK VARCHAR(10) NOT NULL
    #     );
    #     """
    #     if not self.table_exists(table='sp500_list'):
    #         self.execute_query(create_table_query, sp500_list=True)
    #
    #     insert_data_query = f"""
    #     INSERT INTO sp500_list
    #     (stock_index, stock, name, date_added, CIK)
    #     VALUES (%s, %s, %s, %s, %s)
    #     """
    #     # breakpoint()
    #
    #     sql_format_stocks = [[i, s[0], s[1], s[2], s[3]] for i, s in enumerate(stocks)]
    #     self.multiline_query(insert_data_query, sql_format_stocks, sp500_list=True)


    # creates a table in the 'stock_data' database for the ticker parameter
    def create_table(self, ticker, table_suffix='table'):
        ticker = ticker.upper()
        db_ticker = normalize_ticker_name(ticker) # mysql table names cannot contain special character '-'

        create_table_query = f"""
            CREATE TABLE {db_ticker}_{table_suffix} (
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
        if not self.table_exists(ticker=db_ticker):
            self.execute_query(create_table_query)
            print(f'created {db_ticker}_table')


    def create_fundamental_table(self, ticker):
        ticker = ticker.upper()
        db_ticker = normalize_ticker_name(ticker) # mysql table names cannot contain special character '-'

        # key = fiscal_year + time_frame (ex: 2004-annual, or 2010-Q1)
        create_table_query = f"""
            CREATE TABLE {db_ticker}_table (
                my_key VARCHAR (16) PRIMARY KEY,
                fiscal_year INT,
                time_frame VARCHAR (6),
                revenue BIGINT,
                COGS BIGINT,
                gross_income BIGINT,
                SGA BIGINT,
                EBIT BIGINT,
                gross_interest_expense BIGINT,
                pretax_income BIGINT,
                income_tax BIGINT,
                net_income BIGINT,
                shareholder_net_income BIGINT,
                consolidated_net_income BIGINT,
                operating_income BIGINT,
                EPS_basic DECIMAL(6,2),
                EPS_diluted DECIMAL(6,2),
                total_current_assets BIGINT,
                total_noncurrent_assets BIGINT,
                fixed_assets BIGINT,
                total_assets BIGINT,
                total_current_liabilities BIGINT,
                total_noncurrent_liabilities BIGINT,
                total_liabilities BIGINT,
                common_equity BIGINT,
                total_shareholders_equity BIGINT,
                liabilities_and_shareholder_equity BIGINT,
                operating_net_cash_flow BIGINT,
                investing_net_cash_flow BIGINT,
                financing_net_cash_flow BIGINT,
                total_net_cash_flow BIGINT
            );
            """

        # create table in database if it doesn't already exist
        if not self.table_exists(ticker=db_ticker, fundamental_db=True):
            self.execute_query(create_table_query, fundamental_db=True)
            print(f'created {db_ticker}_table')


    # copy new data to SQL, if a table doesn't exist for a stock, create it
    def update_prices_DB_table(self, ticker, stock_data=None):

        if stock_data is None or len(stock_data) == 0:
            return

        db_ticker = normalize_ticker_name(ticker)
        if not self.table_exists(ticker=db_ticker):
            self.create_table(ticker)
            print(f'created {ticker}_table')

        insert_data_query = f"""
            INSERT INTO {db_ticker}_table
            (date, open, high, low, close, adj_close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        self.multiline_query(insert_data_query, stock_data)


    # copy new data to SQL, if a table doesn't exist for a stock, create it
    def update_fundamental_DB_table(self, ticker, stock_data, cik=None):

        # if parse_fundamental_stock_data returns an empty list (no data) then no need to create table
        if not stock_data or len(stock_data) == 0:
            return

        db_ticker = normalize_ticker_name(ticker)
        if not self.table_exists(ticker=db_ticker, fundamental_db=True):
            self.create_fundamental_table(ticker)

        insert_data_query = f"""
            INSERT INTO {db_ticker}_table
            (my_key, fiscal_year, time_frame, revenue, COGS, gross_income, SGA, EBIT, gross_interest_expense, pretax_income, income_tax, net_income, shareholder_net_income, consolidated_net_income, operating_income, EPS_basic, EPS_diluted, total_current_assets, total_noncurrent_assets, fixed_assets, total_assets, total_current_liabilities, total_noncurrent_liabilities, total_liabilities, common_equity, total_shareholders_equity, liabilities_and_shareholder_equity, operating_net_cash_flow, investing_net_cash_flow, financing_net_cash_flow, total_net_cash_flow)
            VALUES ({('%s, ' * len(stock_data[0]))[:-2]})
        """
        # assert(ticker in self.new_stock_data)
        self.multiline_query(insert_data_query, stock_data, fundamental_db=True)

    def query_prices(self, ticker, start_date=None, end_date=None):
        db_ticker = get_db_ticker(ticker)
        data = self._query_table(db_ticker, start_date=start_date, end_date=end_date, fundamental_db=False)

        df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'adj close', 'volume'])
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['adj_close'] = df['close'].astype(float)
        return df

    def query_fundamentals(self, ticker, start_date=None, end_date=None):
        db_ticker = get_db_ticker(ticker)
        data = self._query_table(db_ticker, start_date=start_date, end_date=end_date, fundamental_db=True)
        df = pd.DataFrame(data, columns = ['key', 'fiscal_year', 'time_frame', 'revenue', 'COGS', 'gross_income', 'SGA', 'EBIT', 'gross_interest_expense', 'pretax_income', 'income_tax', 'net_income', 'shareholder_net_income', 'consolidated_net_income', 'operating_income', 'EPS_basic', 'EPS_diluted', 'total_current_assets', 'total_noncurrent_assets', 'fixed_assets', 'total_assets', 'total_current_liabilities', 'total_noncurrent_liabilities', 'total_liabilities', 'common_equity', 'total_shareholders_equity', 'liabilities_and_shareholder_equity', 'operating_net_cash_flow', 'investing_net_cash_flow', 'financing_net_cash_flow', 'total_net_cash_flow'])
        return df

    def _query_table(self, ticker, start_date=None, end_date=None, fundamental_db=False):
        # first check to see if table exists before querying

        start_time = time.time()
        if not self.table_exists(ticker=ticker, fundamental_db=fundamental_db):
            return

        cols = "*"
        if start_date and end_date:
            query = f"SELECT {cols} FROM {ticker}_table WHERE date >= '{start_date}' and date <= '{end_date}'"
        else:
            query = f"SELECT {cols} FROM {ticker}_table"

        data = self.read_query(query, fundamental_db=fundamental_db)

        end_time = time.time()
        # print(f'{end_time - start_time}')
        return data


    def connect_to_mysql_database(self, db_name=None):
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


    # currently uses connection method, faster than mysql connector
    def connect_to_sqlalchemy_database(self, db_name=None):

        db_name = db_name if db_name is not None else self.stock_db
        URI = f'mysql+pymysql://{self.user_name}:{self.user_password}@{self.host_name}:{self.port_number}/{db_name}'

        engine = create_engine(URI)
        connection = engine.connect()

        print("MySQL Database SQL_DB.connection successful")
        return connection


    def create_database(self, query, pw):
        connection = self.connect_to_mysql_database()
        cursor = connection.cursor()

        try:
            cursor.execute(query)
            print("Database created successfully")
        except Error as err:
            # print(f"Error: '{err}'")
            if self.debug:
                breakpoint()
            return -1

        return 0


    def execute_query(self, query, stock_db=True, fundamental_db=False, sp500_list=False):

        if sp500_list:
            db = SQL_DB.sp500_list_connection
        elif fundamental_db:
            db = SQL_DB.fundamental_data_connection
        else:
            db = SQL_DB.stock_data_connection

        cursor = db.cursor()
        try:
            cursor.execute(query)
            db.commit()  # write to db
            # print("Query successful")
        except Error as err:
            print(f"Error: '{err}'")
            if self.debug:
                breakpoint()
            return -1

        return 0


    def multicol_query(self, query, vals, stock_db=True, fundamental_db=False, sp500_list=False):
        if sp500_list:
            db = SQL_DB.sp500_list_connection
        elif fundamental_db:
            db = SQL_DB.fundamental_data_connection
        else:
            db = SQL_DB.stock_data_connection

        cursor = db.cursor()
        try:
            cursor.execute(query, vals)
            db.commit()  # write to db
            # print("Query successful")
        except Error as err:
            print(f"Error: '{err}'")
            if self.debug:
                breakpoint()
            return -1

        return 0


    def multiline_query(self, query, column_values, stock_db=True, fundamental_db=False, sp500_list=False):
        if sp500_list:
            db = SQL_DB.sp500_list_connection
        elif fundamental_db:
            db = SQL_DB.fundamental_data_connection
        else:
            db = SQL_DB.stock_data_connection

        cursor = db.cursor()
        try:
            cursor.executemany(query, column_values)
            db.commit()  # write to db
            # print("Query successful")
        except Error as err:
            print(f"Error: '{err}'")
            if self.debug:
                breakpoint()
            return -1

        return 0


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


    @timer
    def pd_read_query(self, query, stock_db=True, fundamental_db=False, sp500_list=False):

        if fundamental_db:
            db = SQL_DB.read_only_fundamental_data_connection
        else:
            db = SQL_DB.read_only_stock_data_connection

        table_df = pd.read_sql(
            query,
            con=db,
            index_col='date'
        )
        return table_df


    def table_exists(self, ticker=None, table=None, stock_db=True, fundamental_db=False, sp500_list=False) -> bool:
        # must either request ticker table or specific table name
        if not ticker and not table:
            return False

        # if user passes ticker instead of specific table name, generate table name
        if not table:
            table = ticker + '_table'

        if stock_db:
            schema = 'stock_data'
        elif fundamental_db:
            schema = 'fundamental_data'
        else:
            schema = 'stock_list'

        query = f"""
        SELECT EXISTS (
            SELECT TABLE_NAME
            FROM information_schema.TABLES 
            WHERE 
            TABLE_SCHEMA LIKE '{schema}' AND 
                TABLE_TYPE LIKE 'BASE TABLE' AND
                TABLE_NAME = '{table}'
        );
        """

        if sp500_list:
            db = SQL_DB.sp500_list_connection
        elif fundamental_db:
            db = SQL_DB.fundamental_data_connection
        else:
            db = SQL_DB.stock_data_connection

        is_found = self.read_query(query)[0][0]
        return is_found


    def column_exists(self, ticker, col):
        ticker = normalize_ticker_name(ticker)

        query = f"show columns from stock_data.{ticker}_table like '{col}'"
        rst = self.read_query(query)
        return rst != [] and rst != -1


    def write_column(self, db_ticker, data, col_name, sig_digits, after_decimal):
        query = f"""
        ALTER TABLE {db_ticker}_table
        DROP {col_name}
        """
        self.execute_query(query)

        query = f"""
        ALTER TABLE {db_ticker}_table
        ADD {col_name} DECIMAL({sig_digits},{after_decimal})
        """
        self.execute_query(query)

        insert_data_query = f"""
        INSERT INTO {db_ticker}_table
        ({col_name}) 
        VALUES (%s)
        """

        data = self.sql_format_data(data)
        self.multiline_query(insert_data_query, data)


    def write_columns(self, db_ticker, data, col_name, sig_digits, after_decimal):
        for key, val in data.items():
            col = f'{col_name}_{key}'
            query = f"""
            ALTER TABLE {db_ticker}_table
            ADD {col} DECIMAL({sig_digits},{after_decimal});
            """
            data = [[d] for d in data]
            self.multiline_query(query, data)


    def get_latest_sql_date(self, ticker):
        ticker = normalize_ticker_name(ticker) # mysql table names cannot contain special character '-'

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
        ticker = normalize_ticker_name(ticker)

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


    def get_earliest_offset_sql_date(self, ticker, col, offset):
        ticker = normalize_ticker_name(ticker) # mysql table names cannot contain special character '-'

        query = f"""
                SELECT a.{col}
                FROM (
                    SELECT *
                    FROM {ticker}_table
                    order by date asc
                    limit {offset+1}
                ) a
                order by a.date desc
                limit 1
                """
        latest_row = self.read_query(query)
        if latest_row == -1 or latest_row == []:
            return None
        return latest_row[0][0]


    def get_prev_x_rows(self, ticker, col, start_date, x_rows):
        ticker = normalize_ticker_name(ticker)

        query = f"""
                select {col}
                from (
                    select *
                    from stock_data.{ticker}_table
                    where date < '{start_date}'
                    order by date desc
                    limit {x_rows}
                ) a
                order by a.date asc
                """

        query_output = self.read_query(query)
        data = []
        if query_output != -1 and query_output != []:
            for price in query_output:
                data.append(float(price[0]))
        else:
            print(f'error reading from {ticker}_table')
        return data


    def get_latest_non_null_col(self, ticker, col):
        ticker = normalize_ticker_name(ticker)

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
        ticker = normalize_ticker_name(ticker)

        insert_col_query = f"""
        ALTER TABLE stock_data.{ticker}_TABLE
        ADD {col} VARCHAR(10);
        """
        return insert_col_query


    def get_insert_data_query(self, ticker, col):
        ticker = normalize_ticker_name(ticker)

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
        db_ticker = normalize_ticker_name(ticker)

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
        UPDATE {db_ticker}_table
        SET {sel_cols}
        WHERE date=%s
        """
        self.multicol_query(query, vals)


    def delete_table(self, table_name, stock_db=True, fundamental_db=False, sp500_list=False):
        query = f"""
            DROP TABLE {table_name}
        """
        self.execute_query(query, stock_db=stock_db, fundamental_db=fundamental_db, sp500_list=sp500_list)


    # todo - find use case for this function
    # def find_sp500_stock(self, ticker):
    #     i = self.stocks.index(ticker)
    #     return [self.stocks[i]], [self.sp500_date_added[i]], [self.cik_list[i]]


def check_data(financial_statement, key, dec_places=0):
    if key in financial_statement:
        if dec_places == 0:
            data = round(financial_statement[key]['value'])
        else:
            data = round(financial_statement[key]['value'], dec_places)
    else:
        print(f'{key} not in financial statement')
        data = -1
    return data


def normalize_ticker_name(ticker):
    ticker_norm = ticker.replace('-', '').replace('.', '')
    return ticker_norm


if __name__ == "__main__":
    pass

