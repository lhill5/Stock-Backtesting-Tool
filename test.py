import mysql.connector
from mysql.connector import Error
import pandas as pd
import datetime
from os.path import exists
import math
from bs4 import BeautifulSoup
import pandas as pd
import requests
import time

from bs4 import BeautifulSoup
import pandas as pd
import requests
import datetime
from multiprocessing import Pool, cpu_count


def read_stock(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}/history'
    response = requests.get(url)
    html_page = response.content
    soup = BeautifulSoup(html_page, 'html.parser')

    print(ticker)
    ticker_data = []

    latest_sql_date = datetime.date(2021, 5, 28)
    for row in soup.find_all('tr'):
        try:
            date, open_price, high_price, low_price, close_price, adj_close, volume = [r.text for r in row.children]
            date = datetime.datetime.strptime(date, "%b %d, %Y").date()
            if date == latest_sql_date:
                break

            ticker_data.append((date, open_price, high_price, low_price, close_price, adj_close, volume))
        except:
            pass
    return (ticker, ticker_data)


ticker, data = read_stock('AAPL')
new_stock_data = {}
new_stock_data[ticker] = data
cols = "open, high, low, close, adj_close, volume"
cols = cols.split(', ')

insert_data_query = f"""
    INSERT INTO {ticker}_table
    (date, open, high, low, close, adj_close, volume) 
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

df = pd.read_csv(f'stock_data/{ticker}.csv')
df = df.dropna()  # deletes rows with nan in them
column_values = df.values.tolist()
# multiline_query(insert_data_query, column_values)

