from oauth2client.service_account import ServiceAccountCredentials
import gspread
import json
import pandas as pd


SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('../backtrader/stock-data-project-339820-f892045254d5.json', SCOPES)
file = gspread.authorize(creds)
doc = file.open("Stock_Data_Analysis")
# sheet = sheet.sheet_name  #replace sheet_name with the name that corresponds to yours, e.g, it can be sheet1
sheet = doc.worksheet('raw_data')

for data in sheet.range('A1:C2'):
    print(data.value)

df = pd.DataFrame({'col1': [1, 2], 'col2': [5,6]})
sheet.update('A1', [list(df.keys())])
sheet.update('A2', df.values.tolist())

# scopes = [
# 'https://www.googleapis.com/auth/spreadsheets',
# 'https://www.googleapis.com/auth/drive'
# ]
# credentials = ServiceAccountCredentials.from_json_keyfile_name("stock-data-project-339820-f892045254d5.json", scopes) #access the json key you downloaded earlier
# file = gspread.authorize(credentials) # authenticate the JSON key with gspread
# sheet = file.open("Stock_Data_Analysis")  #open sheet
# sheet = sheet.sheet_name  #replace sheet_name with the name that corresponds to yours, e.g, it can be sheet1
#
#
# all_cells = sheet.range('A1:C6')
# print(all_cells)

# import requests
# import config
# import json
#
# account_url = f'{config.ENDPOINT_URL}/v2/account'
# orders_url = f'{config.ENDPOINT_URL}/v2/orders'
# HEADERS = {'APCA-API-KEY-ID': config.APA_API_KEY, 'APCA-API-SECRET-KEY': config.APA_SECRET_KEY}
#
# def get_account():
#     r = requests.get(account_url, headers=HEADERS)
#     return json.loads(r.content)
#
# def create_order(ticker, qty, side, type='market', time_in_force='gtc'):
#     data = {
#         'symbol': ticker,
#         'qty': qty,
#         'side': side,
#         'type': type,
#         'time_in_force': time_in_force
#     }
#     r = requests.get(orders_url, json=data, headers=HEADERS)
#     return json.loads(r.content)
#
# response = create_order('TSLA', 100, 'buy', 'market', 'gtc')
# print(response)
