import requests
import json
from random_functions import convert_str_to_date
import time

api_key = 'SEp3jMzPKmbxBD_UuJw2uYxjfWgqkoaT'


def get_fundamental_data(ticker, fiscal_year=None):
    # url = f'https://api.polygon.io/v2/reference/financials/{ticker}?limit=1&type=Y&apiKey={api_key}'
    # get_latest_url = f'https://api.polygon.io/vX/reference/financials?ticker={ticker.upper()}&include_sources=true&apiKey={api_key}'
    # url = f'https://api.polygon.io/vX/reference/financials?ticker={ticker}&timeframe=annual&order=desc&sort=filing_date&apiKey={api_key}'
    if fiscal_year:
        date_filter_annual_url = f'https://api.polygon.io/vX/reference/financials?ticker={ticker}&filing_date.gte={fiscal_year}-01-01&timeframe=annual&order=asc&limit=1&sort=filing_date&apiKey={api_key}'
    else:
        date_filter_annual_url = f'https://api.polygon.io/vX/reference/financials?ticker={ticker}&filing_date.gte=2000-01-01&timeframe=annual&order=asc&limit=100&sort=filing_date&apiKey={api_key}'

    r = requests.get(date_filter_annual_url).json()

    error = None
    if r['status'] == 'ERROR':
        error = r['error']
        print(error)
        return (None, None)

    results = None
    try:
        results = r['results']
    except:
        breakpoint()

    fundamental_data = {}
    for filing in results:
        start_date = filing['start_date']
        end_date = filing['end_date']
        fiscal_year = convert_str_to_date(start_date).year

        financials = filing['financials']
        balance_sheet = financials['balance_sheet']
        cash_flow_statement = financials['cash_flow_statement']
        comprehensive_income = financials['comprehensive_income']
        income_statement = financials['income_statement']

        fiscal_year_data = {'income_statement': income_statement, 'balance_sheet': balance_sheet, 'cash_flow_statement': cash_flow_statement}
        fundamental_data[fiscal_year] = fiscal_year_data

    return error, fundamental_data


def get_income_statement():
    pass


def get_balance_sheet():
    pass


def get_cash_flow_statement():
    pass


def get_stock_list():
    url = f'https://api.polygon.io/v3/reference/tickers?type=CS&sort=ticker&order=asc&limit=1000&apiKey={api_key}'
    next_url = url

    stock_list = []
    start_time = time.time()
    counter = 0

    while next_url:
        r = requests.get(next_url)
        data = r.json()
        pretty_data = json.dumps(data, indent=4)
        print(pretty_data)
        counter += 1

        # prevent more than 5 API requests per minute
        if counter % 5 == 0:
            while time.time() - start_time < 60:
                pass
            start_time = time.time()

        elif data['status'] == 'OK':
            # add stocks to stocks list
            stock_list.extend(data['results'])

            # if more data, get next url and continue adding stocks
            next_url = None
            if 'next_url' in data:
                next_url = data['next_url']
                next_url += f'&type=CS&active=true&sort=ticker&order=asc&limit=1000&apiKey={api_key}'

        elif data['status'] == 'ERROR':
            print('error processing API request (too many requests)')
            time.sleep(5)

    return stock_list


def json_print(dict):
    json_dict = json.dumps(dict, indent=4)
    print(json_dict)


if __name__ == '__main__':
    ticker = 'AAPL'
    get_fundamental_data(ticker)
    stock_list = get_stock_list()
    with open('stock_list.txt', 'w') as f:
        for stock in stock_list:
            f.write(f'{stock["ticker"]} - active: {stock["active"]}\n')


