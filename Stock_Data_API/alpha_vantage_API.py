import requests
import json

import pandas as pd
from alpha_vantage.timeseries import TimeSeries


# replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
ticker = 'AAPL'
api_key = 'SSFPLDGNSX4NAWV8'

def get_data(ticker):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    return data

def get_company_overview(ticker):
    url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    return data

def get_income_statement(ticker):
    url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&outputsize=full&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    return data

def get_balance_statement(ticker):
    url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&outputsize=full&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    return data

def get_cash_flow_statement(ticker):
    url = f'https://www.alphavantage.co/query?function=CASH_FLOW&outputsize=full&symbol={ticker}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    return data

def get_adjusted_daily(ticker):
    ts = TimeSeries(key='YourAPIkey',  # claim on alphavantage.co
                    output_format='pandas',
                    indexing_type='date')

    daily = ts.get_daily_adjusted(ticker, 'full')[0].sort_index()
    daily.columns = [x[3:] for x in daily.columns]


def calculate_adjusted(df, dividends=False):
    # source_url = 'https://www.tradewithscience.com/stock-split-adjusting-with-python/'

    # we will go from today to the past
    new = df.sort_index(ascending=False)

    split_coef = new['split coefficient'].shift(1
        ).fillna(1).cumprod()

    for col in ['open', 'high', 'low', 'close']:
        new['adj_' + col] = new[col] / split_coef
    new['adj_volume'] = split_coef * new['volume']

    if dividends:
        new['adj_dividends'] = new['dividend amount'] / split_coef

    return new.sort_index(ascending=True)


if __name__ == '__main__':
    income_statement = get_income_statement(ticker)
    latest_income_statement = {key: income_statement[key][0] if key != 'symbol' else income_statement[key] for key in income_statement.keys()}

    income_json = json.dumps(latest_income_statement, indent=4)
    with open('income.txt', 'w') as outfile:
        outfile.write(income_json)


    # balance_statement = get_balance_statement(ticker)
    # latest_balance_statement = {key: balance_statement[key][0] if key != 'symbol' else balance_statement[key] for key in balance_statement.keys()}
    #
    # balance_json = json.dumps(latest_balance_statement, indent=4)
    # with open('balance.txt', 'w') as outfile:
    #     outfile.write(balance_json)


    # cash_flow_statement = get_cash_flow_statement(ticker)
    # latest_cash_flow_statement = {key: cash_flow_statement[key][0] if key != 'symbol' else cash_flow_statement[key] for key in cash_flow_statement.keys()}
    #
    # cash_flow_json = json.dumps(latest_cash_flow_statement, indent=4)
    # with open('cash_flow.txt', 'w') as outfile:
    #     outfile.write(cash_flow_json)

    # url = 'https://www.alphavantage.co/query?function=INCOME_STATEMENT&outputsize=full&symbol=IBM&apikey=demo'
    # r = requests.get(url)
    # data = r.json()
    #
    # for i in data['annualReports']:
    #     print(i)

    # for key, value in data['Time Series (Daily)'].items():
    #     print(key, value)

