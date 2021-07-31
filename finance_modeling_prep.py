import urllib.request
import json


# uses the financial modeling prep API to get stock data (historical / intraday prices, financial analysis (balance sheet /
# class Stock_API:

API_KEY = 'e0ecb6fd8cf50faf985ffe9602bb8fc1'
stock = 'AAPL'
limit = 1
lookup = 'income-statement'
endpoint = f'/api/v3/{lookup}/{stock}'
url = f'https://financialmodelingprep.com{endpoint}?period=quarter&limit=1&apikey={API_KEY}'


# url = 'https://financialmodelingprep.com/api/v3/income-statement/TSLA?period=quarter&limit=400&apikey=e0ecb6fd8cf50faf985ffe9602bb8fc1'
def getResponse(url):
    operUrl = urllib.request.urlopen(url)
    jsonData = None
    if(operUrl.getcode()==200):
        data = operUrl.read()
        jsonData = json.loads(data)
    else:
        print("Error receiving data", operUrl.getcode())

    return jsonData


def main():

    jsonData = getResponse(url)
    # print the state id and state name corresponding

    if jsonData is not None:
        print(jsonData)


if __name__ == '__main__':
    main()

