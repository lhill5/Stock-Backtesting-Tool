from technical_analysis import get_MACD, get_SMA, get_EMA, get_RSI
from collections import defaultdict
from datetime import date, timedelta

from SQL_DB import SQL_DB
from plot_data import Plot



class Stock(SQL_DB):

    def __init__(self, ticker, start_date, end_date):
        super().__init__()

        self.ticker = ticker
        self.ticker_letters = ticker.replace('-', '')
        self.start_date = start_date
        self.end_date = end_date

        self.dates = []
        self.prices = defaultdict(list)
        self.tech_indicators = defaultdict(list)

        self.query_stock_prices()

        dates = self.dates
        self.min_date, self.max_date = dates[0], dates[-1]
        prices = self.prices['close']

        self.date_to_index = get_date_to_index(dates, self.min_date, self.max_date)
        self.index_to_date = get_index_to_date(dates, self.min_date, self.max_date)
        self.unique_indices = list(self.index_to_date.keys())

        # used for slicing dates/prices/etc. lists by index
        self.start_date_index = self.date_to_index[self.start_date]
        self.end_date_index = self.date_to_index[self.end_date]
        self.filtered_unique_indices = self.unique_indices[self.start_date_index:self.end_date_index+1]
        self.filtered_unique_indices = self.unique_indices

        self.get_tech_indicators() # automatically filters results between start/end dates
        self.filter_query_data() # filters dates/prices between start and end date

        plot = Plot(self)
        plot.plot_data()

        # uses a SQL query to get all of the stock's data
        # self.end_date = self.dates[-1]
        # self.start_date = self.end_date - timedelta(days=365)
        # tmp_start_date = trading_days_ago(self.start_date, 200)


        # sets start day a year ago, plus 200 trading days for calculating the SMA/EMA 200 day moving average

        # filter data by last x days
        # index = self.lookup_date_index(self.start_date)
        # self.dates, filtered_data = filter_between(self.dates, self.prices, self.open_price, self.high_price, self.low_price, self.close_price, start_date=tmp_start_date, end_date=self.end_date)
        # self.prices, self.open_price, self.high_price, self.low_price, self.close_price = filtered_data

        # self.MACD_signals = backtest_MACD(dates, prices, self.tech_indicators)
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


    def lookup_date_index(self, target):
        return self.lookup_stock_val(target, self.dates, get_index=True)


    def lookup_price(self, target, get_index=False):
        return self.lookup_stock_val(target, self.prices, get_index)


    # uses binary search to lookup stock val
    def lookup_MACD(self, target, get_index=False):
        return self.lookup_stock_val(target, self.MACD, get_index)


    def lookup_EMA(self, target, EMA_num, get_index=False):
        if EMA_num not in self.EMAs:
            return
        else:
            return self.lookup_stock_val(target, self.EMAs[EMA_num], get_index)


    def lookup_RSI(self, target, get_index=False):
        return self.lookup_stock_val(target, self.RSI, get_index)


    def lookup_stock_val(self, target, arr, get_index=False):
        dates = self.dates

        start = 0
        end = len(dates) - 1
        while start <= end:
            mid = start + ((end - start) // 2)
            # print(type(target), type(dates[mid-1]), type(dates[mid]))

            if dates[mid] == target:
                return mid if get_index else arr[mid]
            # check if target is a weekend and get prev close price
            elif (mid - 1 >= 0 and target > dates[mid - 1] and target < dates[mid]):
                return mid - 1 if get_index else arr[mid - 1]
            elif mid + 1 < len(dates) and target > dates[mid] and target < dates[mid + 1]:
                return mid if get_index else arr[mid]
            elif dates[mid] > target:
                end = mid - 1
            else:
                start = mid + 1


    def query_stock_prices(self, start_date=None, end_date=None):
        cols = "date, open, high, low, close, adj_close, volume"
        if start_date and end_date:
            query = f"SELECT {cols} FROM {self.ticker_letters}_table WHERE date >= {start_date} and date <= {end_date}"
        else:
            query = f"SELECT {cols} FROM {self.ticker_letters}_table"

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
            print(f'error reading from {self.ticker_letters}_table')
            self.get_ticker_data(self.ticker)


    def filter_query_data(self):
        start, end = self.start_date_index, self.end_date_index+1
        self.dates = self.dates[start:end]
        for key in self.prices:
            self.prices[key] = self.prices[key][start:end]


    def query_tech_indicators(self, start_date=None, end_date=None):
        dates, prices = self.dates, self.prices['close']

        cols = "MACD, MACD_signal, MACD_histogram, EMA_9, EMA_50, EMA_150, EMA_200, SMA_9, SMA_50, SMA_150, SMA_200, RSI"
        if start_date and end_date:
            query = f"SELECT {cols} FROM {self.ticker_letters}_table WHERE date >= {start_date} and date <= {end_date} and macd is not null and macd_signal is not null and macd_histogram is not null and EMA_200 is not null and RSI is not null"
        else:
            query = f"SELECT {cols} FROM {self.ticker_letters}_table WHERE macd is not null and macd_signal is not null and macd_histogram is not null and EMA_200 is not null and RSI is not null"

        query_output = self.read_query(query)
        # reads data from sql if columns exist
        latest_date = self.get_latest_sql_date(self.ticker)
        latest_MACD_date = self.get_latest_non_null_col(self.ticker, 'MACD')
        if query_output != -1 and latest_date == latest_MACD_date and latest_MACD_date is not None:
            for row in query_output:
                for ind, val in zip(cols, row):
                    self.tech_indicators[ind].append(float(val))
        else:
            # columns don't exist or data is null, create tables and write to sql db

            # creates columns in sql table
            cols = cols.split(', ')
            for col in cols:
                if not self.column_exists(self.ticker, col):
                    insert_col_query = self.get_new_col_query(self.ticker, col)
                    self.execute_query(insert_col_query)

            # gets technical indicators
            MACD, signal, histogram = get_MACD(dates, prices)
            EMA_9, EMA_50, EMA_150, EMA_200 = [get_EMA(dates, prices, n) for n in [9, 50, 150, 200]]
            SMA_9, SMA_50, SMA_150, SMA_200 = [get_SMA(dates, prices, n) for n in [9, 50, 150, 200]]
            RSI = get_RSI(dates, prices)

            # writes data to table row by row
            for date, MACD_val, signal_val, histogram_val, EMA_9_val, EMA_50_val, EMA_150_val, EMA_200_val, SMA_9_val, SMA_50_val, SMA_150_val, SMA_200_val, RSI_val in zip(dates, MACD, signal, histogram, EMA_9, EMA_50, EMA_150, EMA_200, SMA_9, SMA_50, SMA_150, SMA_200, RSI):
                self.update_row(date, cols, (MACD_val, signal_val, histogram_val, EMA_9_val, EMA_50_val, EMA_150_val, EMA_200_val, SMA_9_val, SMA_50_val, SMA_150_val, SMA_200_val, RSI_val, date))

            # stores technical indicator data instead of re-reading sql query
            indicators = [MACD, signal, histogram, EMA_9, EMA_50, EMA_150, EMA_200, SMA_9, SMA_50, SMA_150, SMA_200, RSI]
            for ind, val in zip(cols, indicators):
                self.tech_indicators[ind] = val


        # cols = "MACD, signal, histogram, EMA_9, EMA_50, EMA_150, EMA_200, SMA_9, SMA_50, SMA_150, SMA_200, RSI"
        # for col in cols.split(', '):
        #     if start_date and end_date:
        #         query = f"SELECT {col} FROM {self.ticker}_table WHERE date >= {start_date} and date <= {end_date}"
        #     else:
        #         query = f"SELECT {col} FROM {self.ticker}_table"
        #
        #     rows = self.read_query(query)
        #     col_exists = rows != -1
        #     latest_date = self.get_latest_sql_date(self.ticker)
        #     latest_col_date = self.get_latest_non_null_col(self.ticker, col)
        #     # if read query was successful (column exists) then read data
        #     if col_exists and latest_date == latest_col_date and latest_date is not None:
        #         data = [data[0] for data in rows]
        #         self.tech_indicators[col] = data
        #     elif col not in ['signal', 'histogram']:
        #         # if technical indicator doesn't exist in the sql table, calculate it and write to sql
        #         tech_ind = get_tech_indicator(col, self.dates, self.prices['close'])
        #         if col == 'MACD':
        #             MACD, signal, histogram = tech_ind
        #             MACD, signal, histogram = self.sql_format_data(MACD), self.sql_format_data(signal), self.sql_format_data(histogram)
        #             data = [MACD, signal, histogram]
        #         else:
        #             data = [self.sql_format_data(tech_ind)]
        #
        #
        #         # write data to sql so we don't have to recalculate this every time
        #         insert_cols = ['MACD', 'signal', 'histogram'] if col == 'MACD' else [col]
        #         for i, insert_col in enumerate(insert_cols):
        #             if not col_exists:
        #                 insert_col_query = self.get_new_col_query(col)
        #                 self.execute_query(insert_col_query)
        #
        #             insert_data_query = self.get_insert_data_query(col)
        #             self.multiline_query(insert_data_query, data[i])
        #
        #             self.tech_indicators[col] = data


    def get_tech_indicators(self):
        dates = self.dates
        prices = self.prices['close']
        moving_averages = [9,50,150,200]
        start, end = self.start_date_index, self.end_date_index+1

        MACD, signal, histogram = get_MACD(prices[start - 33: end])
        EMAs = {n:get_EMA(prices[start - n: end], n+1) for n in moving_averages}
        SMAs = {n: get_SMA(prices[start - n: end], n+1) for n in moving_averages}
        RSI = get_RSI(prices[start - 15: end])

        self.tech_indicators['MACD'] = MACD
        self.tech_indicators['signal'] = signal
        self.tech_indicators['histogram'] = histogram
        self.tech_indicators['RSI'] = RSI

        self.tech_indicators['EMA'] = EMAs
        self.tech_indicators['SMA'] = SMAs


    def get_column(self, column_name):
        # not tested
        query = f"SELECT {column_name} FROM stock_list"
        data = self.read_query(query)
        data = [d[0] for d in data]
        return data


    def write_column(self, data, col_name, sig_digits, after_decimal):
        query = f"""
        ALTER TABLE {self.ticker_letters}_table
        DROP {col_name}
        """
        self.execute_query(query)

        query = f"""
        ALTER TABLE {self.ticker_letters}_table
        ADD {col_name} DECIMAL({sig_digits},{after_decimal})
        """
        self.execute_query(query)

        insert_data_query = f"""
        INSERT INTO {self.ticker_letters}_table
        ({col_name}) 
        VALUES (%s)
        """

        data = self.sql_format_data(data)
        self.multiline_query(insert_data_query, data)


    def write_columns(self, data, col_name, sig_digits, after_decimal):
        for key, val in data.items():
            col = f'{col_name}_{key}'
            query = f"""
            ALTER TABLE {self.ticker_letters}_table
            ADD {col} DECIMAL({sig_digits},{after_decimal});
            """
            data = [[d] for d in data]
            self.multiline_query(query, data)


    def update_row(self, row_id, cols, vals):

        vals_list = list(vals)
        for i, val in enumerate(vals_list):
            vals_list[i] = val if val is None or isinstance(val, date) else round(val, 5)
        vals = tuple(vals_list)

        sel_cols = ""
        for i, col in enumerate(cols):
            sel_cols += f'{col}=%s'
            if col != cols[-1]:
                sel_cols += ', '

        query = f"""
        UPDATE {self.ticker_letters}_table
        SET {sel_cols}
        WHERE date=%s
        """
        self.multicol_query(query, vals)


    def delete_columns(self, cols, ticker):
        cols = cols.split(', ')
        ticker = ticker.replace('-', '')
        for col in cols:
            query = f"""
            ALTER TABLE stock_data.{ticker}_TABLE
            DROP {col}
            ;
            """
            self.execute_query(query)


def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)
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

