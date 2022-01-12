import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import requests
from urllib.request import urlretrieve as retrieve
import random
import os
import pandas as pd
import datetime


def get_historic_data(ticker, reused=False):

    ticker.replace('.', '-')
    ticker = ticker.upper()

    # date = datetime.datetime.fromtimestamp(83548800)
    url = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=0&period2=1622332800&interval=1d&events=history&includeAdjustedClose=true'
    # check if webiste exists or not
    request = requests.get(url)

    if request.status_code == 200:
        if reused:
            ticker = ticker[:-1]
        retrieve(url, f'stock_data/{ticker}.csv')
    elif request.status_code == 404:
        if reused:
            return
        elif len(ticker) > 2 and ticker[-2:] == '-U':
            get_historic_data(ticker.replace("-U", "-UN"), reused=True)


# def get_all_stock_data():
#     tickers = pd.read_csv('../stock_list.csv', names=['ticker symbols'])
#     i = 1
#     for col, stock_name in tickers.iterrows():
#         ticker, name, *_ = stock_name[0].split(' - ')
#         ticker = ticker.replace('.', '-')  # brk.b -> brk-b (format yahoo-finance uses
#         print(ticker)
#         # i += 1
#         # if i != 80:
#         #     continue
#
#         # only get stock data if csv doesn't exist
#         if not os.path.isfile(f'../stock_data/{ticker}.csv'):
#             # wait random time between 1-2 seconds before scrapping from website (so we don't get kicked from website)
#             wait_interval = round(random.uniform(1, 3), 1)
#             sleep(wait_interval)
#             get_stock_data(ticker)


# wait for set period of time before webscrapping next ticker
def webscrape_ticker(ticker):
    # only get stock data if csv doesn't exist

    if not os.path.isfile(f'../stock_data/{ticker}.csv'):
        # wait random time between 1-2 seconds before scrapping from website (so we don't get kicked from website)
        wait_interval = round(random.uniform(1, 3), 1)
        sleep(wait_interval)
        get_historic_data(ticker)


# scrap_yahoo_stock_data('aapl')
# selenium_get_stock_date('aapl')

if __name__ == '__main__':
    get_historic_data('BITE-U')



##############################################

# def selenium_get_stock_date(ticker):
#
#     ticker.replace('.', '-')
#     url = f'https://finance.yahoo.com/quote/{ticker}/history'
#
#     driver = webdriver.Chrome('/Users/landon/Documents/python/chromedriver')
#     driver.get(url)
#
#     # date_range = WebDriverWait(driver, 20).until(EC.visibility_of_all_elements_located((By.XPATH, "//div[contains(@class, 'dateRangeBtn')]")))
#     date_range = driver.find_element_by_xpath("//div[contains(@class, 'dateRangeBtn')]")
#     date_range.click()
#     print('clicked on data range')
#
#     max_buttom = driver.find_element_by_xpath("//button[contains(@data-value, 'MAX')]")
#     max_buttom.click()
#     print('clicked on max data')
#
#     csv_name = f'{ticker.upper()}.csv'
#     download_link = driver.find_element_by_xpath(f"//a[@download='{csv_name}']")
#
#     print(download_link)
#     download_link.click()
#     print('downloaded raw data')
#
#     sleep(10)
#     driver.close()
#
#
# def scrap_yahoo_stock_data(ticker):
#     if len(ticker) < 1:
#         return
#
#     ticker.replace('.', '-')
#     url = f'https://finance.yahoo.com/quote/{ticker}/history'
#
#     website = requests.get(url)
#     soup = scrap(website.content, 'html.parser')
#     table_tag = soup.find('table')
#
#     rows = table_tag.find_all('tr')
#     print(len(rows))
#     columns, rows = rows[0], rows[1:]
#
#     stock_dict = {}
#     keys = [col.text.lower() for col in columns.find_all('th')]
#     # for row in rows:
#     #     x = row.find_all('td')
#
#     for i, key in enumerate(keys):
#         stock_dict[key] = [row.find_all('td')[i].text for row in rows if len(row.find_all('td')) == 7]
#
#     dates = stock_dict['date']
#     print(dates[0], dates[-1])
