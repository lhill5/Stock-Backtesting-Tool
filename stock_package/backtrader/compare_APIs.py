from Stock_Prices import StockPrices
from Stock_Fundamentals import StockFundamentals
from SQL_DB import SQL_DB
import requests
import datetime
import pandas as pd
import random
import time

POLYGON_API_KEY = 'SEp3jMzPKmbxBD_UuJw2uYxjfWgqkoaT'
ALPHA_VANTAGE_API_KEY = 'SSFPLDGNSX4NAWV8'


def compare_data(df1, df2):
    # adj close is only available in df1 (from SQL tables) not df2
    if df1.date[0] > df2.date[0]:
        dates = df1.date
    else:
        dates = df2.date

    df1 = df1.drop(['adj_close'], axis=1)

    # only compare data which overlaps between df1 and df2
    start_date = max(df1.date[0], df2.date[0], datetime.date(2017,1,1))
    end_date = min(df1.date[len(df1.date)-1], df2.date[len(df2.date)-1])

    # filter data between start/end dates
    df1 = df1.loc[(df1.date >= start_date) & (df1.date <= end_date)]
    df2 = df2.loc[(df2.date >= start_date) & (df2.date <= end_date)]

    orig_df1 = df1.copy()
    orig_df2 = df2.copy()

    # if after filtering indexing is from 1000-2000, then now its 0-1000
    df1.reset_index(inplace = True)
    df2.reset_index(inplace = True)

    df1 = df1.round(1)
    df2 = df2.round(1)

    df1 = df1.drop(['index', 'volume'], axis=1)
    df2 = df2.drop(['index', 'volume'], axis=1)
    # df1 = df1.drop(['index'], axis=1)
    # df2 = df2.drop(['index'], axis=1)

    cmp_df = df1.compare(df2)
    cmp_df['date'] = [dates[i] for i in cmp_df.index]
    return cmp_df


def getPolygonData(ticker):
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/1980-01-01/2022-02-26?sort=asc&limit=50000&apiKey={POLYGON_API_KEY}'
    r = requests.get(url).json()

    if 'results' not in r:
        print('reached polygon API limit, returning ...')
        return None

    df = pd.DataFrame(r['results'])
    # re-format timestamp column from milliseconds since epoch to timestamp
    df.t = df.t.apply(lambda t: (datetime.datetime.fromtimestamp(t / 1000)).date())
    # delete unused columns
    df = df.drop(['vw', 'n'], axis=1)

    df = df.rename(columns={'v': 'volume', 'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 't': 'date'})
    df = df.reindex(columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop(['volume'], axis=1)
    return df


def getAlphaVantageData(ticker):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}'

    r = requests.get(url)
    r = r.json()

    data_key = 'Time Series (Daily)'
    if data_key not in r:
        print('reached alpha vantage API limit, returning ...')
        return None

    data = r[data_key]
    df = pd.DataFrame(data).T
    df = df.iloc[::-1] # reverse order of df (ascending order)

    df['date'] = df.index
    df.index = list(range(len(data)))
    df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'})
    df = df.reindex(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    # change data types
    df['date'] = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in df['date']]
    df['open'] = df['open'].astype('float64')
    df['high'] = df['high'].astype('float64')
    df['low'] = df['low'].astype('float64')
    df['close'] = df['close'].astype('float64')
    df['volume'] = df['volume'].astype('int32')

    return df


def getAlphaVantageIntraDay(ticker):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=5min&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}'

    r = requests.get(url)
    r = r.json()

    data_key = 'Time Series (5min)'
    if data_key not in r:
        print('reached alpha vantage API limit, returning ...')
        return None

    data = r[data_key]
    df = pd.DataFrame(data).T
    return df


def getAlphaVantageWeeklyData(ticker):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}'

    r = requests.get(url)
    r = r.json()

    data_key = 'Weekly Adjusted Time Series'
    if data_key not in r:
        print('reached alpha vantage API limit, returning ...')
        return None

    data = r[data_key]
    df = pd.DataFrame(data).T
    df = df.iloc[::-1] # reverse order of df (ascending order)
    df_copy = df.copy()

    df['date'] = df.index
    df.index = list(range(len(data)))
    df = df.rename(columns={'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. volume': 'volume'})
    df = df.reindex(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

    # change data types
    df['date'] = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in df['date']]
    df['open'] = df['open'].astype('float64')
    df['high'] = df['high'].astype('float64')
    df['low'] = df['low'].astype('float64')
    df['close'] = df['close'].astype('float64')
    df = df.drop(['volume'], axis=1)
    # df['volume'] = df['volume'].astype('int32')

    return df


def myprint(data):
    df = data
    # prints all rows/cols
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3):
        print(df)


ticker = 'AAPL'
database = SQL_DB(None, update=False)
ticker_prices = Prices_API(ticker, database)
ticker_fundamental = Fundamental_API(ticker, database)

sp500_tickers = database.stocks
stocks_only = database.read_stock_list(stocks_only=True)
random.shuffle(stocks_only)

start_time = time.time()
for ticker in ['AAPL', 'XTNT', 'VBNK', 'FBRX']:
# for ticker in stocks_only:
    print(f'comparing {ticker}')
    sql_data = Prices_API(ticker, database).data
    if sql_data is None:
        print('no sql data')
        continue

    poly_data = getPolygonData(ticker)
    alpha_data = getAlphaVantageData(ticker)
    # alpha_intraday = getAlphaVantageIntraDay(ticker)
    # alpha_weekly_data = getAlphaVantageWeeklyData(ticker)

    df1 = sql_data
    df2 = alpha_data

    if df1.empty:
        print('sql data empty')
    if df2.empty:
        print('alpha vantage data empty')

    if not df1.empty and not df2.empty:
        cmp_df = compare_data(df1, df2)
        print(cmp_df)

    print('hello')
