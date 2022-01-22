import requests
import config
import json

account_url = f'{config.ENDPOINT_URL}/v2/account'
orders_url = f'{config.ENDPOINT_URL}/v2/orders'
HEADERS = {'APCA-API-KEY-ID': config.APA_API_KEY, 'APCA-API-SECRET-KEY': config.APA_SECRET_KEY}

def get_account():
    r = requests.get(account_url, headers=HEADERS)
    return json.loads(r.content)

def create_order(ticker, qty, side, type='market', time_in_force='gtc'):
    data = {
        'symbol': ticker,
        'qty': qty,
        'side': side,
        'type': type,
        'time_in_force': time_in_force
    }
    r = requests.get(orders_url, json=data, headers=HEADERS)
    return json.loads(r.content)

response = create_order('TSLA', 100, 'buy', 'market', 'gtc')
print(response)
