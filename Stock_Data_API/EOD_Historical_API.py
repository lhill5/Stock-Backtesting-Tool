import requests

EOD_API_KEY = '621c4bcb9a8ea0.20478455'
import requests

import pandas as pd

from io import StringIO

def get_eod_data(ticker="AAPL", api_token=EOD_API_KEY, session=None):

    symbol = ticker + '.US'
    if session is None:
        session = requests.Session()

        url = 'https://eodhistoricaldata.com/api/eod/%s' % symbol
        params = {"api_token": api_token, 'from': '2017-01-01', 'to': '2022-01-01'}
        r = session.get(url, params=params)

        if r.status_code == requests.codes.ok:
            df = pd.read_csv(StringIO(r.text), skipfooter=1, parse_dates=[0], index_col=0, engine='python')
            return df
        else:
            raise Exception(r.status_code, r.reason, url)


def myprint(data):
    df = data
    # prints all rows/cols
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 3):
        print(df)


df = get_eod_data(ticker='AAPL')
print(df)