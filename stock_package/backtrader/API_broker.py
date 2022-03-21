import requests
import json
import config
import pandas as pd


BASE_URL = 'https://paper-api.alpaca.markets'
ORDERS_URL = f'{BASE_URL}/v2/orders'
ACCOUNT_URL = f'{BASE_URL}/v2/account'
HEADERS = {'APCA-API-KEY-ID': config.APA_API_KEY, 'APCA-API-SECRET-KEY': config.APA_SECRET_KEY}


# response = create_order('AAPL', 100, 'buy', 'market', 'gtc')
# print(response)


class Broker_API:
    def __init__(self):
        NY = 'America/New_York'
        start=pd.Timestamp('2020-08-01', tz=NY).isoformat()
        end=pd.Timestamp('2020-08-30', tz=NY).isoformat()

    def submit_market_buy_order(self, symbol, qty):
        data = {
            'symbol': symbol,
            'qty': qty,
            'side': 'buy',
            'type': 'market',
            'time_in_force': 'gtc'
        }
        r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
        response = json.loads(r.content)

        # error checking, forbidden means symbol isn't valid
        if 'status' in response and response['status'] == 'accepted':
            print(f'fulfilled buy order for {qty} share{"s" if qty != 1 else ""} of {symbol}')
            return response
        elif 'code' in response and str(response['code'])[0] == '4':
            print(response['message'])
            return
        else:
            print('buy order not fulfilled, unknown error')


    def submit_market_sell_order(self, symbol, qty):
        data = {
            'symbol': symbol,
            'qty': qty,
            'side': 'sell',
            'type': 'market',
            'time_in_force': 'gtc'
        }
        r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
        response = json.loads(r.content)

        # error checking, forbidden means symbol isn't valid
        if 'status' in response and response['status'] == 'accepted':
            print(f'fulfilled sell order for {qty} share{"s" if qty != 1 else ""} of {symbol}')
            return response
        elif 'code' in response and str(response['code'])[0] == '4':
            print(response['message'])
            return
        else:
            print('sell order not fulfilled, unknown error')


    def get_account(self):
        r = requests.get(ACCOUNT_URL, headers=HEADERS)
        return json.loads(r.content)

    def get_orders(self):
        r = requests.get(ORDERS_URL, headers=HEADERS)
        return json.loads(r.content)

    # Get only the closed orders for a particular stock or all stocks
    def get_closed_positions(self, ticker=None):
        pass

    # Get only the closed orders for a particular stock
    def get_open_positions(self, ticker=None):
        pass

    def get_assets(self):
        pass

    def get_daily_stock_data(self, stock):
        pass

if __name__ == '__main__':
    broker = Broker_API()
    # broker.submit_market_buy_order('AAPL', 1)
    # broker.submit_market_sell_order('AAPL', 100)
