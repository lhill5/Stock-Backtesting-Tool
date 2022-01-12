import requests
import json
import time


api_key = 'SEp3jMzPKmbxBD_UuJw2uYxjfWgqkoaT'

def get_stock_list():
    url = f'https://api.polygon.io/v3/reference/tickers?type=CS&active=true&sort=ticker&order=asc&limit=1000&apiKey={api_key}'
    next_url = url

    stocks = []
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
            stocks.extend(data['results'])

            # if more data, get next url and continue adding stocks
            next_url = None
            if 'next_url' in data:
                next_url = data['next_url']
                next_url += f'&type=CS&active=true&sort=ticker&order=asc&limit=1000&apiKey={api_key}'

        elif data['status'] == 'ERROR':
            print('error processing API request (too many requests)')
            time.sleep(5)


if __name__ == '__main__':
    pass

