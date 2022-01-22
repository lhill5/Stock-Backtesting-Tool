import alpaca_trade_api as tradeapi
import pandas as pd
import config
import sys


class Broker_API:
    def __init__(self):
        NY = 'America/New_York'
        start=pd.Timestamp('2020-08-01', tz=NY).isoformat()
        end=pd.Timestamp('2020-08-30', tz=NY).isoformat()

        self.api = tradeapi.REST(key_id=config.APA_API_KEY, secret_key=config.APA_SECRET_KEY, api_version='v2')


    def submit_market_order(self, stock, qty):
        # Submit a market order to buy 1 share of Apple at market price
        self.api.submit_order(
            symbol=stock,
            qty=qty,
            side='buy',
            type='market'
            # time_in_force='gtc'
        )


    # Get only the closed orders for a particular stock or all stocks
    def get_closed_positions(self, ticker=None):
        # Get the last 100 of our closed orders
        closed_orders = self.api.list_orders(
            status='closed',
            limit=100,
            nested=True  # show nested multi-leg orders
        )
        if ticker is not None:
            closed_orders = [o for o in closed_orders if o.symbol == ticker]

        return closed_orders


    # Get only the closed orders for a particular stock
    def get_open_positions(self, ticker=None):
        # Get the last 100 of our closed orders
        open_orders = self.api.list_orders(
            status='open',
            limit=100,
            nested=True  # show nested multi-leg orders
        )
        if ticker is not None:
            open_orders = [o for o in open_orders if o.symbol == ticker]

        return open_orders


    def get_assets(self):
        active_assets = self.api.list_assets(status='active')
        tickers = [(a.symbol, a.exchange) for a in active_assets]
        return tickers


    def get_daily_stock_data(self, stock):
        NY = 'America/New_York'
        start = pd.Timestamp('2016-01-01', tz=NY).isoformat()
        end = pd.Timestamp('2021-08-14', tz=NY).isoformat()
        stock_data = self.api.get_barset(stock, '1D', start=start, end=end).df
        print(stock_data.index.values)
        print(stock_data)

        rows = []
        for d in stock_data:
            date = d.t.date()
            rows.append((date, d.o, d.h, d.l, d.c, d.v))

        return rows


if __name__ == '__main__':
    pass
