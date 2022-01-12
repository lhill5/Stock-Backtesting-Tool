from financial_calcs import get_last_trading_day
import mysql.connector
from mysql.connector import Error
import math
import time
import datetime
import pdb
from urllib.error import HTTPError
import fundamentalData_API as fundamental_API


class SQL_DB:
    stock_data_connection = None
    fundamental_data_connection = None

    def __init__(self, stock_data_API, update=False, update_list=None):

        self.start_time = time.time()

        # update_list = ['MIK']
        self.stock_data_API = stock_data_API

        self.host_name = "localhost"
        self.user_name = "root"
        self.user_password = 'mssqlserver44'
        self.stock_db = 'stock_data'
        self.fundamental_db = 'fundamental_data'

        self.update = update
        self.update_list = update_list
        self.debug = False
        self.daily_API_limit_reached = False

        if not SQL_DB.stock_data_connection:
            SQL_DB.stock_data_connection = self.connect_to_database(self.stock_db)
        if not SQL_DB.fundamental_data_connection:
            SQL_DB.fundamental_data_connection = self.connect_to_database(self.fundamental_db)

        # self.update_stock_list()

        # self.stocks = self.stock_data_API.get_stock_list()
        # self.stocks = self.read_stock_list()[14700:14720]
        if self.update_list is not None:
            self.stocks = self.update_list
        else:
            self.stocks = self.read_stock_list(stocks_only=True)

        # print(f'{len(self.stocks)} stocks in sql db')

        self.latest_trading_date = get_last_trading_day()  # - datetime.timedelta(days=1)
        if self.update:

            # self.latest_stock_date = {ticker: self.get_latest_sql_date(ticker) for ticker in self.stocks}

            # self.update_stocks = [stock for stock in self.stocks if self.latest_stock_date[stock] is None or self.latest_stock_date[stock] < self.latest_trading_date]
            self.update_stocks = self.stocks
            self.new_stock_data = {}

            print(f'{len(self.stocks)} tables, update {len(self.update_stocks)} tables')
            self.sql_update_tables(update_prices=False)


    def get_SQL_stock_connection(self):
        return SQL_DB.stock_data_connection


    # creates a fact table of all stocks currently being used
    def create_stock_list_table(self):

        stocks = self.stock_data_API.get_stock_list()

        create_table_query = f"""
        CREATE TABLE stock_list (
            stock_index INT PRIMARY KEY,
            stock VARCHAR (16) NOT NULL,
            exchange VARCHAR (32) NOT NULL
        );
        """
        if not self.table_exists('stock_list'):
            self.execute_query(create_table_query)

        insert_data_query = f"""
        INSERT INTO stock_list
        (stock_index, stock, exchange)
        VALUES (%s, %s, %s)
        """

        sql_format_stocks = [[i, s[0], s[1]] for i, s in enumerate(stocks)]
        self.multiline_query(insert_data_query, sql_format_stocks)


    def create_stock_only_list_table(self):

        stocks = self.stock_data_API.get_stock_only_list()

        create_table_query = f"""
        CREATE TABLE stock_only_list (
            stock_index INT PRIMARY KEY,
            stock VARCHAR (16) NOT NULL,
            exchange VARCHAR (32) NOT NULL
        );
        """
        if not self.table_exists('stock_only_list'):
            self.execute_query(create_table_query)

        insert_data_query = f"""
        INSERT INTO stock_only_list
        (stock_index, stock, exchange)
        VALUES (%s, %s, %s)
        """

        sql_format_stocks = [[i, s[0], s[1]] for i, s in enumerate(stocks)]
        self.multiline_query(insert_data_query, sql_format_stocks)


    # creates a table in the 'stock_data' database for the ticker parameter
    def create_table(self, ticker, table_suffix='table'):
        ticker = ticker.upper()
        norm_ticker = normalize_ticker_name(ticker) # mysql table names cannot contain special character '-'

        create_table_query = f"""
            CREATE TABLE {norm_ticker}_{table_suffix} (
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
        if not self.table_exists(f'{norm_ticker}_table'):
            self.execute_query(create_table_query)
            print(f'created {norm_ticker}_table')


    def create_fundamental_table(self, ticker):
        ticker = ticker.upper()
        norm_ticker = normalize_ticker_name(ticker) # mysql table names cannot contain special character '-'

        # key = fiscal_year + time_frame (ex: 2004-annual, or 2010-Q1)
        create_table_query = f"""
            CREATE TABLE {norm_ticker}_table (
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
        if not self.table_exists(f'{norm_ticker}_table', fundamental_db=True):
            self.execute_query(create_table_query, fundamental_db=True)
            print(f'created {norm_ticker}_table')


    def sql_update_tables(self, update_prices=True, update_fundamental=True):
        # self.update_stocks = self.update_stocks[:20]
        # ______________________________________________________________________________________________________________
        stock_data_counter = 0
        fundamental_data_counter = 0

        stock_data_queue = []
        fundamental_data_queue = []

        start_time = time.time()
        update_list_len = len(self.update_stocks)
        for i, ticker in enumerate(self.update_stocks):
            norm_ticker = normalize_ticker_name(ticker)
            # if ticker == 'AADI':
            #     breakpoint()

            # get stock prices data
            if update_prices:
                # check to see if data already exists before API call and SQL table edit
                prices_data = self.query_table(norm_ticker)
                if prices_data:
                    pass

                if stock_data_counter % 300 == 0 and stock_data_counter != 0:
                    if time.time() - start_time >= 60:
                        start_time = time.time()
                        stock_data_queue.append({norm_ticker: self.get_stock_data(ticker)})
                        stock_data_counter += 1
                    # if still waiting, update SQL table in the meantime
                    else:
                        if stock_data_queue:
                            queue_data = stock_data_queue.pop()
                            norm_ticker, data = list(queue_data.items())[0]
                            if data:
                                self.update_stock_prices_table(norm_ticker, data)
                            else:
                                print(f'{norm_ticker} no data')
                else:
                    stock_data_queue.append({norm_ticker: self.get_stock_data(ticker)})
                    stock_data_counter += 1

            # get fundamental data
            if update_fundamental:
                # api requires ticker be capitalized
                fundamental_data = self.query_table(norm_ticker, fundamental_db=True)
                if fundamental_data:
                    pass

                norm_ticker = norm_ticker.upper()
                if fundamental_data_counter % 5 == 0 and fundamental_data_counter != 0:
                    # check if 1 minute (API limit) has passed before another API call, if last element then call update_fundamental_table
                    if time.time() - start_time >= 60:
                        start_time = time.time()
                        fundamental_data_queue.append({norm_ticker: self.parse_fundamental_stock_data(ticker)})
                        fundamental_data_counter += 1
                    # if still waiting, update SQL table in the meantime
                    else:
                        if fundamental_data_queue:
                            data = None
                            try:
                                queue_data = fundamental_data_queue.pop()
                                norm_ticker, data = list(queue_data.items())[0]
                            except:
                                breakpoint()

                            if data:
                                self.update_fundamental_table(norm_ticker, data)
                            else:
                                print(f'{norm_ticker} no data')
                else:
                    fundamental_data_queue.append({norm_ticker: self.parse_fundamental_stock_data(ticker)})
                    fundamental_data_counter += 1

        # update remaining stocks in queue if any left
        if update_prices:
            while len(stock_data_queue) != 0:
                queue_data = stock_data_queue.pop()
                norm_ticker, data = list(queue_data.items())[0]
                if data:
                    self.update_stock_prices_table(norm_ticker, data)
                else:
                    print(f'{norm_ticker} no data')
        if update_fundamental:
            while len(fundamental_data_queue) != 0:
                queue_data = fundamental_data_queue.pop()
                norm_ticker, data = list(queue_data.items())[0]
                if data:
                    self.update_fundamental_table(norm_ticker, data)
                else:
                    print(f'{norm_ticker} no data')


    def get_stock_data(self, ticker, similar_ticker=None):
        norm_ticker = normalize_ticker_name(ticker)
        # print(ticker)

        # if no data exists for ticker, then use an arbitrary start date to get all data for that ticker
        latest_stock_date = self.get_latest_sql_date(ticker)
        if latest_stock_date is None:
            start_date = datetime.date(1900,1,1)
        else:
            start_date = latest_stock_date + datetime.timedelta(days=1)
        end_date = self.latest_trading_date

        try:
            # row_data = []
            # uses financial modeling prep API to get historical stock data
            row_data = self.stock_data_API.get_historical_daily(ticker, start_date, end_date)
        except HTTPError:
            self.daily_API_limit_reached = True
            print('API rate limit reached.')
            return (ticker, [])
        return (ticker, row_data)


    def parse_fundamental_stock_data(self, ticker):
        norm_ticker = normalize_ticker_name(ticker)
        error, fundamental_data = fundamental_API.get_fundamental_data(ticker)
        if fundamental_data is None:
            return

        # transform data to list of tuples (for inserting into SQL table)

        transformed_data = []
        for fiscal_year, data in fundamental_data.items():
            income_statement = data['income_statement']
            revenue = check_data(income_statement, 'revenues')
            COGS = check_data(income_statement, 'cost_of_revenue')
            gross_income = check_data(income_statement, 'gross_profit')
            SGA = check_data(income_statement, 'operating_expenses')
            EBIT = check_data(income_statement, 'operating_income_loss')
            gross_interest_expense = check_data(income_statement, 'interest_expense_operating')
            pretax_income = check_data(income_statement, 'income_loss_from_continuing_operations_before_tax')
            income_tax = check_data(income_statement, 'income_tax_expense_benefit')
            net_income = check_data(income_statement, 'net_income_loss')
            shareholder_net_income = check_data(income_statement, 'net_income_loss_available_to_common_stockholders_basic')
            consolidated_net_income = check_data(income_statement, 'net_income_loss_attributable_to_parent')
            operating_income = check_data(income_statement, 'operating_income_loss')
            EPS_basic = check_data(income_statement, 'basic_earnings_per_share', dec_places=2)
            EPS_diluted = check_data(income_statement, 'diluted_earnings_per_share', dec_places=2)

            balance_sheet = data['balance_sheet']
            total_current_assets = check_data(balance_sheet, 'current_assets')
            total_noncurrent_assets = check_data(balance_sheet, 'noncurrent_assets')
            fixed_assets = check_data(balance_sheet, 'fixed_assets')
            total_assets = check_data(balance_sheet, 'assets')
            total_current_liabilities = check_data(balance_sheet, 'current_liabilities')
            total_noncurrent_liabilities = check_data(balance_sheet, 'noncurrent_liabilities')
            total_liabilities = check_data(balance_sheet, 'liabilities')
            common_equity = check_data(balance_sheet, 'equity_attributable_to_parent')
            total_shareholders_equity = check_data(balance_sheet, 'equity')
            total_equity = check_data(balance_sheet, 'equity')
            liabilities_and_shareholder_equity = check_data(balance_sheet, 'liabilities_and_equity')

            cashflow = data['cash_flow_statement']
            investing_net_cash_flow = check_data(cashflow, 'net_cash_flow_from_investing_activities')
            financing_net_cash_flow = check_data(cashflow, 'net_cash_flow_from_financing_activities')
            operating_net_cash_flow = check_data(cashflow, 'net_cash_flow_from_operating_activities')
            total_net_cash_flow = check_data(cashflow, 'net_cash_flow')

            time_frame = 'annual'
            key = f'{fiscal_year}-{time_frame}'
            transformed_data.append((key, fiscal_year, time_frame, revenue, COGS, gross_income, SGA, EBIT, gross_interest_expense, pretax_income, income_tax, net_income, shareholder_net_income, consolidated_net_income, operating_income, EPS_basic, EPS_diluted, total_current_assets, total_noncurrent_assets, fixed_assets, total_assets, total_current_liabilities, total_noncurrent_liabilities, total_liabilities, common_equity, total_shareholders_equity, liabilities_and_shareholder_equity, operating_net_cash_flow, investing_net_cash_flow, financing_net_cash_flow, total_net_cash_flow))

        return transformed_data


    def log_result(self, result):

        ticker, data = result
        self.new_stock_data[ticker] = data


    # copy new data to SQL, if a table doesn't exist for a stock, create it
    def update_stock_prices_table(self, ticker, stock_data=None):
        if self.daily_API_limit_reached:
            return

        if stock_data is None:
            ticker, stock_data = self.get_stock_data(ticker)

        norm_ticker = normalize_ticker_name(ticker)
        if not self.table_exists(f'{norm_ticker}_table'):
            self.create_table(ticker)
            print(f'created {ticker}_table')

        insert_data_query = f"""
            INSERT INTO {norm_ticker}_table
            (date, open, high, low, close, adj_close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        if stock_data is not None and len(stock_data) > 0:
            self.multiline_query(insert_data_query, stock_data)


    def update_fundamental_table(self, ticker, stock_data=None):
        if self.daily_API_limit_reached:
            return

        if stock_data is None:
            stock_data = self.parse_fundamental_stock_data(ticker)

        # if parse_fundamental_stock_data returns an empty list (no data) then no need to create table
        if not stock_data:
            return

        norm_ticker = normalize_ticker_name(ticker)
        if not self.table_exists(f'{norm_ticker}_table', fundamental_db=True):
            self.create_fundamental_table(ticker)

        insert_data_query = f"""
            INSERT INTO {norm_ticker}_table
            (my_key, fiscal_year, time_frame, revenue, COGS, gross_income, SGA, EBIT, gross_interest_expense, pretax_income, income_tax, net_income, shareholder_net_income, consolidated_net_income, operating_income, EPS_basic, EPS_diluted, total_current_assets, total_noncurrent_assets, fixed_assets, total_assets, total_current_liabilities, total_noncurrent_liabilities, total_liabilities, common_equity, total_shareholders_equity, liabilities_and_shareholder_equity, operating_net_cash_flow, investing_net_cash_flow, financing_net_cash_flow, total_net_cash_flow)
            VALUES ({('%s, ' * len(stock_data[0]))[:-2]})
        """

        # if ticker not in self.new_stock_data:
        #     print(ticker, self.new_stock_data)
        #     if self.debug:
        #         pdb.set_trace()

        # assert(ticker in self.new_stock_data)
        # stock_data = self.new_stock_data[ticker]
        if stock_data is not None and len(stock_data) > 0:
            self.multiline_query(insert_data_query, stock_data, fundamental_db=True)


    def query_table(self, ticker, start_date=None, end_date=None, fundamental_db=False):
        # first check to see if table exists before querying
        if not self.table_exists(f'{ticker}_table', fundamental_db=fundamental_db):
            return

        cols = "*"
        if start_date and end_date:
            query = f"SELECT {cols} FROM {ticker}_table WHERE date >= '{start_date}' and date <= '{end_date}'"
        else:
            query = f"SELECT {cols} FROM {ticker}_table"

        data = self.read_query(query, fundamental_db=fundamental_db)
        return data


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


    def create_database(self, query, pw):
        connection = self.connect_to_database()
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


    def execute_query(self, query, fundamental_db=False):
        if fundamental_db:
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


    def multicol_query(self, query, vals, fundamental_db=False):
        if fundamental_db:
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


    def multiline_query(self, query, column_values, fundamental_db=False):
        if fundamental_db:
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


    def read_query(self, query, fundamental_db=False):
        if fundamental_db:
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


    def table_exists(self, table_name, fundamental_db=False):
        query = f"SELECT * FROM {table_name}"
        if fundamental_db:
            db = SQL_DB.fundamental_data_connection
        else:
            db = SQL_DB.stock_data_connection

        cursor = db.cursor(buffered=True)
        try:
            cursor.execute(query)
            return 1
        except Error as err:
            if self.debug:
                breakpoint()
            return 0


    def column_exists(self, ticker, col):
        ticker = normalize_ticker_name(ticker)

        query = f"show columns from stock_data.{ticker}_table like '{col}'"
        rst = self.read_query(query)
        return rst != [] and rst != -1


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
        norm_ticker = normalize_ticker_name(ticker)

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
        UPDATE {norm_ticker}_table
        SET {sel_cols}
        WHERE date=%s
        """
        self.multicol_query(query, vals)


    def delete_table(self, table_name):
        query = f"""
            DROP TABLE {table_name}
        """
        self.execute_query(query)


    def update_stock_list(self):
        # self.delete_table('stock_list')
        # self.create_stock_list_table()
        self.delete_table('stock_only_list')
        self.create_stock_only_list_table()


    def read_stock_list(self, stocks_only=False, US_only=False):
        if US_only:
            query = "SELECT stock FROM stock_list where exchange like '%NYSE%' or exchange like '%NASDAQ%'"
        elif stocks_only:
            query = "SELECT stock FROM stock_only_list"
        else:
            query = "SELECT stock FROM stock_list"

        stocks = self.read_query(query)
        stocks = [stock[0] for stock in stocks]
        return stocks


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

