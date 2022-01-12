import requests

ticker = 'AAPL'
api_key = ''
url = f"https://www.quandl.com/api/v3/databases/{ticker}/data?download_type=full&api_key={api_key}"

r = requests.get(url)
data = r.json()
print(data)

