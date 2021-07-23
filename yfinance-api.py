import yfinance as yf
from backtrader.SQL_DB import SQL_DB
from multiprocessing import Pool, cpu_count
import time

missing = 0
found = 0


def read_stock(ticker):

    tickerData = yf.Ticker(ticker)
    #get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start='2021-07-01', end='2021-07-03', auto_adjust=False)
    return (ticker, tickerDf)


def get_stock_data(ticker, similar_ticker=None):
    print(ticker)

    # get data from yfinance API
    tickerData = yf.Ticker(ticker) if not similar_ticker else yf.Ticker(similar_ticker)
    # get the historical prices for this ticker
    tickerDf = tickerData.history(period='1d', start='2021-07-01', end='2021-07-10', auto_adjust=False)

    # transform data into list of tuples, where each tuple represents a single day's amount of data (date, open, high, low, close, adj_close, volume)
    dates = [d.date() for d in tickerDf['Open'].keys()]
    open_prices, high_prices, low_prices, close_price, adj_close, volume, dividends, splits = [list(tickerDf[k]) for k in tickerDf.keys()]
    row_data = list(zip(dates, open_prices, high_prices, low_prices, close_price, adj_close, volume))

    if len(tickerDf['Open']) == 0 and len(ticker) > 2 and ticker[-2:] == '-U' and similar_ticker is None:
        return get_stock_data(ticker, ticker + 'N')
    else:
        return (ticker, row_data)


def log_data(stock_data):
    global missing, found

    ticker, prices = stock_data
    row_count = len(prices)
    if row_count != 2:
        missing += 1
        print(f'missing {missing} stocks, found {found} stocks')
    else:
        found += 1


def main():

    db = SQL_DB()
    print(len(db.stocks), db.stocks)
    pool = Pool(cpu_count())
    try:
        for ticker in db.stocks:
            pool.apply_async(get_stock_data, args=(ticker,None), callback=log_data)
            # data = read_stock(ticker)
            # log_data(data)
        pool.close()
        pool.join()
    except ValueError as e:
        print(e)

    print(f'found {found} stocks, {missing} missing stocks')


if __name__ == '__main__':
    # start timer to see how long program took to finish
    start_time = time.time()

    # main()
    ticker, data = read_stock('FVIV')
    dates = [d.date() for d in data['Open'].keys()]
    open_prices, high_prices, low_prices, close_price, adj_close, volume = [list(data[k]) for k in data.keys()[:6]]

    row_data = zip(dates, open_prices, high_prices, low_prices, close_price, adj_close, volume)
    for row in row_data:
        print(row)

    end_time = time.time()
    print(f'program finished in {round(end_time - start_time, 1)} seconds')

