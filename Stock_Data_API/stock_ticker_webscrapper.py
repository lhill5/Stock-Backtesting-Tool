from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import csv


PATH = "/Users/landon/Downloads/chromedriver"
op = webdriver.ChromeOptions()
op.add_argument('headless')
driver = webdriver.Chrome(PATH)

# driver.get("https://stockanalysis.com/stocks/")
# stock_ticker_list = driver.find_element_by_class_name('no-spacing')
# stock_tickers_elem = stock_ticker_list.find_elements_by_tag_name('li')
# print(len(stock_tickers_elem))
# stock_tickers = []
# for stock in stock_tickers_elem:
#     a_tag = stock.find_element_by_tag_name('a')
#     stock_tickers.append(a_tag.text)

# driver.get('https://stockanalysis.com/etf/')
# etf_ticker_list = driver.find_element_by_class_name('no-spacing')
# etf_tickers_elem = etf_ticker_list.find_elements_by_tag_name('li')
# print(len(etf_tickers_elem))
# etf_tickers = [etf.text for etf in etf_tickers_elem]

# # lookup s&p 500 stocks
driver.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
sp500_ticker_list = driver.find_element_by_id('constituents')
sp500_tickers_elem = sp500_ticker_list.find_elements_by_tag_name('tr')

sp500_tickers = []
for ticker in sp500_tickers_elem:
    if ticker.find_element_by_xpath('..').tag_name == "tbody":
        td = ticker.find_elements_by_tag_name('td')
        symbol, name, date_added, cik  = td[0].text, td[1].text, td[6].text, td[7].text
        print(symbol)
        sp500_tickers.append(symbol + ' - ' + name + ' - ' + date_added + ' - ' + cik)

# ________________________________________________________
# stock_count = len(stock_tickers)
# etf_count = len(etf_tickers)

# print(type(etf_tickers))
# print(f'{stock_count} Stock ticker symbols exist')
# print(f'{etf_count} ETF ticker symbols exist')
# ________________________________________________________


# stock_tickers.extend(etf_tickers)
# # # create csv file for all stock symbols so this program doesn't have to be re-run each time
# with open('stock_list.csv', 'w') as stocklist:
#     filewriter = csv.writer(stocklist, delimiter=',')
#     for stock in stock_tickers:
#         filewriter.writerow([stock])


with open('s&p_500.csv', 'w') as sp500_list:
    filewriter = csv.writer(sp500_list, delimiter=',')
    for stock in sp500_tickers:
        print(stock)
        filewriter.writerow([stock])


driver.quit()

