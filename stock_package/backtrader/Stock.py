from technical_analysis import *
from collections import defaultdict
import datetime
import pdb
import time
import connectorx as cx


class Stock:

    def __init__(self, ticker, SQL, start_date=None, end_date=None):
        # pdb.set_trace()
        # super().__init__()
        # print(ticker)

        self.ticker = ticker
        self.SQL = SQL
        self.ticker_letters = ticker.replace('-', '')
        self.error = False

        self.dates = []
        self.prices = defaultdict(list)
        self.fundamental = defaultdict(list)
        self.tech_indicators = defaultdict(list)
        self.moving_averages = [9,50,150,200]

        start_time = time.time()
        # update stock if no data
        earliest_date = self.SQL.get_earliest_sql_date(self.ticker)
        if earliest_date is None:
            self.SQL.update_table(self.ticker)

        # gets min/max dates from SQL table, used to ensure user doesn't request date outside of these date ranges
        self.valid_start_date = earliest_date
        self.valid_end_date = self.SQL.get_latest_sql_date(self.ticker)
        end_time = time.time()
        # print(f'sql query program finished in {round((end_time - start_time), 1)} seconds')


        # start_time = time.time()
        # for i in range(10000):
        #     self.valid_start_date = self.SQL.get_earliest_sql_date(self.ticker)
        # end_time = time.time()
        # print(f'sql query program finished in {round(end_time - start_time, 1)} seconds')

        # if stock still has no data, move onto next stock
        if self.valid_start_date is None or self.valid_end_date is None:
            return

        # if user tries querying between invalid start/end date (don't have data between these time periods) then move onto next stock
        if not (start_date >= self.valid_start_date and end_date <= self.valid_end_date):
            return

        # gets an offset start date in order to calculate technical indicators that need 'x' number of previous rows for calculation
        # start_time = time.time()
        self.valid_start_date_offset = self.SQL.get_earliest_offset_sql_date(self.ticker, 'date', 200)
        # end_time = time.time()
        # print(f'sql query program finished in {round((end_time - start_time)*1000, 1)} ms')

        # if not enough data to calculate EMA 200 (200 day offset) then move onto next stock
        if self.valid_start_date_offset >= self.valid_end_date:
            return

        # sets start/end date. If user didn't enter start/end date then show all data. If invalid input, set to valid dates based on above variables
        self.start_date = self.valid_start_date_offset if start_date is None else max(start_date, self.valid_start_date)
        self.end_date = self.valid_end_date if end_date is None else min(end_date, self.valid_end_date)

        error = self.query_stock_prices()
        if error == -1:
            self.error = True
            return

        rst = self.get_tech_indicators()  # automatically filters results between start/end dates

        # if not enough stock data to calculate the technical indicators, move onto next stock

        if rst == 0:
            print("couldn't calculate tech indicators")
            return

        self.date_to_index = get_date_to_index(self.dates, self.start_date, self.end_date)
        self.index_to_date = get_index_to_date(self.dates, self.start_date, self.end_date)
        self.unique_indices = list(self.index_to_date.keys())

        # used for slicing dates/prices/etc. lists by index
        self.start_date_index = self.date_to_index[self.start_date]
        self.end_date_index = self.date_to_index[self.end_date]
        self.filtered_unique_indices = self.unique_indices[self.start_date_index:self.end_date_index+1]
        self.filtered_unique_indices = self.unique_indices

        self.stock_dict = {'date': [str(d) for d in self.dates], 'open': self.prices['open'], 'high': self.prices['high'], 'low': self.prices['low'], 'close': self.prices['close'], 'adj_close': self.prices['adj_close'], 'volume': self.prices['volume']}

        # uses a SQL query to get all of the stock's data
        # self.end_date = self.dates[-1]
        # self.start_date = self.end_date - timedelta(days=365)
        # tmp_start_date = trading_days_ago(self.start_date, 200)


        # sets start day a year ago, plus 200 trading days for calculating the SMA/EMA 200 day moving average

        # filter data by last x days
        # index = self.lookup_date_index(self.start_date)
        # self.dates, filtered_data = filter_between(self.dates, self.prices, self.open_price, self.high_price, self.low_price, self.close_price, start_date=tmp_start_date, end_date=self.end_date)
        # self.prices, self.open_price, self.high_price, self.low_price, self.close_price = filtered_data

        # self.MACD_signals = x(dates, prices, self.tech_indicators)
        # self.EMA_signals = backtest_EMA(dates, prices, self.tech_indicators)
        # self.RSI_signals = backtest_RSI(dates, prices, self.tech_indicators)

        # self.dates, filtered_data = filter_between(self.dates, self.prices, self.open_price, self.high_price, self.low_price, self.close_price, self.MACD, self.signal, self.histogram, self.RSI, self.SMAs[9], self.SMAs[50], self.SMAs[150], self.SMAs[200], self.EMAs[9], self.EMAs[50], self.EMAs[150], self.EMAs[200], start_date=self.start_date, end_date=self.end_date)
        # self.prices, self.open_price, self.high_price, self.low_price, self.close_price, self.MACD, self.signal, self.histogram, self.RSI, self.SMAs[9], self.SMAs[50], self.SMAs[150], self.SMAs[200], self.EMAs[9], self.EMAs[50], self.EMAs[150], self.EMAs[200] = filtered_data

        # self.MACD_signals = filter_trade_signals(self.MACD_signals, start_date=self.start_date, end_date=self.end_date)
        # self.EMA_signals = filter_trade_signals(self.EMA_signals, start_date=self.start_date, end_date=self.end_date)
        # self.RSI_signals = filter_trade_signals(self.RSI_signals, start_date=self.start_date, end_date=self.end_date)

        # self.date_to_index = {d: i for i, d in enumerate(self.dates)}
        # self.index = [dt.date2num(d) for d in self.dates]
        # if len(self.dates) != 0:
        #     self.min_date, self.max_date = self.dates[0], self.dates[-1]
        #
        # self.percent_change = pd.Series(self.adj_close_price).pct_change()


    def query_stock_prices(self):
        cols = "date, open, high, low, close, adj_close, volume"
        query = f"SELECT {cols} FROM {self.ticker_letters}_table WHERE date >= '{self.start_date}' and date <= '{self.end_date}'"
        # query = f"SELECT {cols} FROM {self.ticker_letters}_table limit 1"

        query_output = self.SQL.read_query(query)
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
            print(f'error reading from {self.ticker_letters}_table')

        if len(self.prices['open']) == 0:
            return -1


    def filter_query_data(self):
        start, end = self.start_date_index, self.end_date_index+1
        self.dates = self.dates[start:end]
        for key in self.prices:
            self.prices[key] = self.prices[key][start:end]


    # def query_tech_indicators(self, start_date=None, end_date=None):
    #     dates, prices = self.dates, self.prices['close']
    #
    #     cols = "MACD, MACD_signal, MACD_histogram, EMA_9, EMA_50, EMA_150, EMA_200, SMA_9, SMA_50, SMA_150, SMA_200, RSI"
    #     if start_date and end_date:
    #         query = f"SELECT {cols} FROM {self.ticker_letters}_table WHERE date >= {start_date} and date <= {end_date} and macd is not null and macd_signal is not null and macd_histogram is not null and EMA_200 is not null and RSI is not null"
    #     else:
    #         query = f"SELECT {cols} FROM {self.ticker_letters}_table WHERE macd is not null and macd_signal is not null and macd_histogram is not null and EMA_200 is not null and RSI is not null"
    #
    #     query_output = self.read_query(query)
    #     # reads data from sql if columns exist
    #     latest_date = self.get_latest_sql_date(self.ticker)
    #     latest_MACD_date = self.get_latest_non_null_col(self.ticker, 'MACD')
    #     if query_output != -1 and latest_date == latest_MACD_date and latest_MACD_date is not None:
    #         for row in query_output:
    #             for ind, val in zip(cols, row):
    #                 self.tech_indicators[ind].append(float(val))
    #     else:
    #         # columns don't exist or data is null, create tables and write to sql db
    #
    #         # creates columns in sql table
    #         cols = cols.split(', ')
    #         for col in cols:
    #             if not self.column_exists(self.ticker, col):
    #                 insert_col_query = self.get_new_col_query(self.ticker, col)
    #                 self.execute_query(insert_col_query)
    #
    #         # gets technical indicators
    #         MACD, signal, histogram = get_MACD(prices)
    #         EMA_9, EMA_50, EMA_150, EMA_200 = [get_EMA(prices, n) for n in [9, 50, 150, 200]]
    #         SMA_9, SMA_50, SMA_150, SMA_200 = [get_SMA(prices, n) for n in [9, 50, 150, 200]]
    #         RSI = get_RSI(prices)
    #
    #         # writes data to table row by row
    #         for date, MACD_val, signal_val, histogram_val, EMA_9_val, EMA_50_val, EMA_150_val, EMA_200_val, SMA_9_val, SMA_50_val, SMA_150_val, SMA_200_val, RSI_val in zip(dates, MACD, signal, histogram, EMA_9, EMA_50, EMA_150, EMA_200, SMA_9, SMA_50, SMA_150, SMA_200, RSI):
    #             self.update_row(date, cols, (MACD_val, signal_val, histogram_val, EMA_9_val, EMA_50_val, EMA_150_val, EMA_200_val, SMA_9_val, SMA_50_val, SMA_150_val, SMA_200_val, RSI_val, date))
    #
    #         # stores technical indicator data instead of re-reading sql query
    #         indicators = [MACD, signal, histogram, EMA_9, EMA_50, EMA_150, EMA_200, SMA_9, SMA_50, SMA_150, SMA_200, RSI]
    #         for ind, val in zip(cols, indicators):
    #             self.tech_indicators[ind] = val


    def get_tech_indicators(self):

        x_days = max(self.moving_averages)
        # see how long this takes to run
        prev_close_prices = self.SQL.get_prev_x_rows(self.ticker, 'close', self.start_date, x_days)
        prev_high_prices = self.SQL.get_prev_x_rows(self.ticker, 'high', self.start_date, x_days)
        prev_low_prices = self.SQL.get_prev_x_rows(self.ticker, 'low', self.start_date, x_days)

        curr_close_prices = self.prices['close']
        curr_high_prices = self.prices['high']
        curr_low_prices = self.prices['low']

        close_prices = prev_close_prices + curr_close_prices
        high_prices = prev_high_prices + curr_high_prices
        low_prices = prev_low_prices + curr_low_prices

        if len(close_prices) != (len(curr_close_prices) + x_days):
            return 0

        start = len(prev_close_prices)
        MACD, signal, histogram = get_MACD(close_prices[start - 33:])
        EMAs = {n:get_EMA(close_prices[start - n:], n+1) for n in self.moving_averages}
        SMAs = {n: get_SMA(close_prices[start - n:], n+1) for n in self.moving_averages}
        RSI = get_RSI(close_prices[start - 15:])

        neg_DI14s, pos_DI14s, ADXs = get_ADX(high_prices[start-27:], low_prices[start-27:], close_prices[start-27:])
        self.tech_indicators['MACD'] = MACD
        self.tech_indicators['signal'] = signal
        self.tech_indicators['histogram'] = histogram
        self.tech_indicators['RSI'] = RSI

        self.tech_indicators['EMA'] = EMAs
        self.tech_indicators['SMA'] = SMAs

        self.tech_indicators['-DI'] = neg_DI14s
        self.tech_indicators['+DI'] = pos_DI14s
        self.tech_indicators['ADX'] = ADXs

        self.fundamental_data = self.SQL.query_table(self.ticker, fundamental_db=True)
        return 1


    def get_column(self, column_name):
        # not tested
        query = f"SELECT {column_name} FROM stock_list"
        data = self.SQL.read_query(query)
        data = [d[0] for d in data]
        return data


    def write_column(self, data, col_name, sig_digits, after_decimal):
        query = f"""
        ALTER TABLE {self.ticker_letters}_table
        DROP {col_name}
        """
        self.SQL.execute_query(query)

        query = f"""
        ALTER TABLE {self.ticker_letters}_table
        ADD {col_name} DECIMAL({sig_digits},{after_decimal})
        """
        self.SQL.execute_query(query)

        insert_data_query = f"""
        INSERT INTO {self.ticker_letters}_table
        ({col_name}) 
        VALUES (%s)
        """

        data = self.SQL.sql_format_data(data)
        self.SQL.multiline_query(insert_data_query, data)


    def write_columns(self, data, col_name, sig_digits, after_decimal):
        for key, val in data.items():
            col = f'{col_name}_{key}'
            query = f"""
            ALTER TABLE {self.ticker_letters}_table
            ADD {col} DECIMAL({sig_digits},{after_decimal});
            """
            data = [[d] for d in data]
            self.SQL.multiline_query(query, data)


    # def update_row(self, row_id, cols, vals):
    #
    #     vals_list = list(vals)
    #     for i, val in enumerate(vals_list):
    #         vals_list[i] = val if val is None or isinstance(val, datetime.date) else round(val, 5)
    #     vals = tuple(vals_list)
    #
    #     sel_cols = ""
    #     for i, col in enumerate(cols):
    #         sel_cols += f'{col}=%s'
    #         if col != cols[-1]:
    #             sel_cols += ', '
    #
    #     query = f"""
    #     UPDATE {self.ticker_letters}_table
    #     SET {sel_cols}
    #     WHERE date=%s
    #     """
    #     self.SQL.multicol_query(query, vals)


    def delete_columns(self, cols, ticker):
        cols = cols.split(', ')
        ticker = ticker.replace('-', '')
        for col in cols:
            query = f"""
            ALTER TABLE stock_data.{ticker}_TABLE
            DROP {col}
            ;
            """
            self.SQL.execute_query(query)


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + datetime.timedelta(n)
    # dates_with_weekends = []
    # for d in dates:
    #     dates_with_weekends.append(d)
    #     if d.weekday() == 4:
    #         dates_with_weekends.append(d + timedelta(days=1))
    #         dates_with_weekends.append(d + timedelta(days=2))
    #
    # for d in dates_with_weekends:
    #     yield d


# returns dictionary where key=date, value=index
def get_date_to_index(dates, min_date, max_date):
    i = -1
    date_dict = {d: 0 for d in dates}

    date_to_index = {}
    for d in daterange(min_date, max_date):
        if d.weekday() < 5 and d in date_dict:
            i += 1
        date_to_index[d] = i
    return date_to_index


# returns dictionary where key=index, value=date
def get_index_to_date(dates, min_date, max_date):
    i = -1
    date_dict = {d: 0 for d in dates}

    index_to_date = {}
    for d in daterange(min_date, max_date):
        if d.weekday() < 5 and d in date_dict:
            i += 1
            index_to_date[i] = d
    return index_to_date


if __name__ == '__main__':
    pass
