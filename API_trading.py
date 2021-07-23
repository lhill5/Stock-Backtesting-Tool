import alpaca_trade_api as tradeapi
import requests
from backtrader import config

daily_bars_url = config.BARS_URL + '/1D?symbols=AACE'
r = requests.get(daily_bars_url, headers=config.HEADERS)
# print(json.dumps(r.json(), indent=4))

api = tradeapi.REST(
    'PK384SOJP5WX4PEU6VLA',
    'xKzwLzv8eyDc3s9JuUdY7uyjk3CZHlny9nALROFJ',
    'https://paper-api.alpaca.markets', api_version='v2'
)

active_assets = api.list_assets(status='active')
# Filter the assets down to just those on NASDAQ.
tickers = [a.symbol for a in active_assets]
tickers = [a for a in active_assets]
for t in tickers:
    if t.symbol == 'AFBI':
        print(t)

print(len(tickers))
# print(tickers)



# Submit a market order to buy 1 share of Apple at market price
# api.submit_order(
#     symbol='AAPL',
#     qty=1,
#     side='buy',
#     type='market'
#     # time_in_force='gtc'
# )


# Get a list of all of our positions.
portfolio = api.list_positions()

# Print the quantity of shares for each position.
# for position in portfolio:
#     print(f"{position.qty} shares of {position.symbol}")

