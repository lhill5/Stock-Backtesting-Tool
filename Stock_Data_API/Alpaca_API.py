import alpaca_trade_api as tradeapi
from stock_package.backtrader import config
import pandas as pd


NY = 'America/New_York'
start=pd.Timestamp('2020-08-01', tz=NY).isoformat()
end=pd.Timestamp('2020-08-30', tz=NY).isoformat()
# print(api.get_barset(['AAPL', 'GOOG'], 'minute', start=start, end=end).df)


api = tradeapi.REST(key_id=config.API_KEY, secret_key=config.SECRET_KEY, api_version='v2')


def get_assets():
    active_assets = api.list_assets(status='active')
    tickers = [(a.symbol, a.exchange) for a in active_assets]
    return tickers


def get_daily_stock_data(stock):
    NY = 'America/New_York'
    start = pd.Timestamp('2016-01-01', tz=NY).isoformat()
    end = pd.Timestamp('2021-08-14', tz=NY).isoformat()
    stock_data = api.get_barset(stock, '1D', start=start, end=end).df
    print(stock_data.index.values)
    print(stock_data)

    # rows = []
    # for d in stock_data:
    #     date = d.t.date()
    #     rows.append((date, d.o, d.h, d.l, d.c, d.v))


    # print(rows[0])
    # See how much stock moved in that timeframe.


def submit_market_order(stock, qty):
# Submit a market order to buy 1 share of Apple at market price
    api.submit_order(
        symbol=stock,
        qty=qty,
        side='buy',
        type='market'
        # time_in_force='gtc'
    )


get_daily_stock_data('EMP')

