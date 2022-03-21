from tech_indicators import *
from collections import defaultdict
import datetime

from Stock_Prices import StockPrices
from Stock_Fundamentals import StockFundamentals
from global_functions import get_db_ticker
from global_decorators import *


# @timer
class Stock:

    def __init__(self, ticker, SQL, logger, market_calendar, request_start_date=None, request_end_date=None):
        start = time.time()
        self.ticker = ticker
        self.SQL = SQL
        self.logger = logger
        self.market_calendar = market_calendar

        self.db_ticker = get_db_ticker(ticker)
        self.error = False

        self.valid_start_date, self.valid_end_date = self.getValidDates(request_start_date, request_end_date)
        if self.valid_start_date >= self.valid_end_date or (self.valid_end_date - self.valid_start_date).days < 100:
            self.logger.error(f'Not enough data points to graph {self.ticker} stock')
            return

        # uses separate classes designed to get stock data either through SQL or an API if not available
        self.prices_API = StockPrices(self.SQL)
        self.fundamental_API = StockFundamentals(self.SQL)

        # get price data
        price_data = self.prices_API.get_data(self.ticker, start_date=self.valid_start_date, end_date=self.valid_end_date)
        self.dates = price_data['date'].values
        self.dates = price_data['date'].values
        self.open = price_data['open'].values
        self.high = price_data['high'].values
        self.low = price_data['low'].values
        self.close = price_data['close'].values
        self.adj_close = price_data['adj_close'].values
        self.volume = price_data['volume'].values
        self.date_ohlcv = {'date': self.dates, 'open': self.open, 'high': self.high, 'low': self.low, 'close': self.close, 'adj close': self.adj_close, 'volume': self.volume}
        self.ohlcv = {'open': self.open, 'high': self.high, 'low': self.low, 'close': self.close, 'adj close': self.adj_close, 'volume': self.volume}

        # get fundamental data
        self.fundamentals = self.fundamental_API.get_data(self.ticker)

        self.tech_indicators = defaultdict(list)
        self.moving_averages = [9,50,150,200]

        rst = self.get_tech_indicators()
        if rst == 0:
            # if not enough stock data to calculate the technical indicators, move onto next stock
            self.logger.error(f"couldn't calculate tech indicators for {self.db_ticker}")
            return

        # used for fast lookup from date to index or vice versa, typically used in Graph class when hovering over certain day/index
        self.date_to_index = get_date_to_index(self.dates, self.valid_start_date, self.valid_end_date)
        self.index_to_date = get_index_to_date(self.dates, self.valid_start_date, self.valid_end_date)

        end = time.time()
        print(f'stock init time: {end - start}')


    # todo - add to tech_indicators file
    def get_tech_indicators(self):

        x_days = max(self.moving_averages)
        # see how long this takes to run
        prev_close_prices = self.SQL.get_prev_x_rows(self.ticker, 'close', self.valid_start_date, x_days)
        prev_high_prices = self.SQL.get_prev_x_rows(self.ticker, 'high', self.valid_start_date, x_days)
        prev_low_prices = self.SQL.get_prev_x_rows(self.ticker, 'low', self.valid_start_date, x_days)

        curr_close_prices = self.close.tolist()
        curr_high_prices = self.high.tolist()
        curr_low_prices = self.low.tolist()

        close_prices = prev_close_prices + curr_close_prices
        high_prices = prev_high_prices + curr_high_prices
        low_prices = prev_low_prices + curr_low_prices

        # -- find --
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

        self.MACD = {'MACD': MACD, 'signal': signal}
        self.RSI = {'RSI': RSI}
        self.ADX = {'ADX': ADXs, '+DI': pos_DI14s, '-DI': neg_DI14s}
        return 1


    # determines what the start and end date can be based on data available and requested dates
    def getValidDates(self, request_start_date, request_end_date):
        # update stock if no data
        data_earliest_date = self.SQL.get_earliest_sql_date(self.ticker)
        data_latest_date = self.SQL.get_latest_sql_date(self.ticker)

        if data_earliest_date is None:
            return

        if request_start_date is None:
            request_start_date = data_earliest_date
        if request_end_date is None:
            request_end_date = data_latest_date

        # get valid start date
        earliest_date_index = self.market_calendar.date_lookup[data_earliest_date]
        earliest_date_offset = self.market_calendar.index_lookup[earliest_date_index + 200]
        if data_earliest_date < request_start_date:
            if earliest_date_offset <= request_start_date:
                valid_start_date = request_start_date
            else:
                valid_start_date = earliest_date_offset
        else:
            valid_start_date = earliest_date_offset

        # get valid end date
        if request_end_date > data_latest_date:
            valid_end_date = data_latest_date
        else:
            valid_end_date = request_end_date

        return valid_start_date, valid_end_date


    # todo - below functions should be a part of SQL_DB
    def get_column(self, column_name):
        # not tested
        query = f"SELECT {column_name} FROM stock_list"
        data = self.SQL.read_query(query)
        data = [d[0] for d in data]
        return data


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
    #     UPDATE {self.db_ticker}_table
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
# @timer
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
