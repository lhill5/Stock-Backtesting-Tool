from Stock import Stock
from Graph import Graph
from SQL_DB import SQL_DB
import time
import datetime
import pandas as pd


start_time = time.time()
database = SQL_DB(update=False)

start_date, end_date = datetime.date(2020,1,1), datetime.date(2021,1,1)
# create Stock objects to plot/screen from
ticker = 'MSFT'

stock = Stock('msft', start_date=datetime.date(2020,1,1), end_date=datetime.date(2021,7,11))
MSFT = stock.stock_dict
df = pd.DataFrame(MSFT)

fig = Graph(ticker, stock.stock_dict, stock.tech_indicators, stock.date_to_index)
fig.plot()

        # query_rst = database.read_query(query)
        # dates, open_price, high_price, low_price, close_price, adj_close_price, volume = query_rst
        # if len(database.dates) >= 200:
        #     stock = Stock(ticker, dates, prices, open_price, high_price, low_price, close_price, volume)
        #     stock_lookup[ticker] = stock
            # plot = Plot(stock)
            # plot.stock_plot()

    # filtered_stocks = Stock_Screener(stock_lookup)
    # filtered_stocks.print()

    # print(stock_ data.keys())

# stock_data = data['data']
# dates, prices, open_price, high_price, low_price, close_price = stock_data['dates'], stock_data['prices'], stock_data['open'], stock_data['high'], stock_data['low'], stock_data['close']
# stock = Stock(ticker, dates, prices, open_price, high_price, low_price, close_price)

end_time = time.time()
print(f'program finished in {round(end_time - start_time, 1)} seconds')
