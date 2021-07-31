from urllib.request import urlopen
import json

def get_jsonparsed_data(url):
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

url = ("https://financialmodelingprep.com/api/v3/quote/AMZN?apikey=e0ecb6fd8cf50faf985ffe9602bb8fc1")
print(get_jsonparsed_data(url))

