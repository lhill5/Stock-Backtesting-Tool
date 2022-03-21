# To run application:
#   bokeh serve --log-file backtrader/log_bokeh.log --show backtrader
#   bokeh serve --show backtrader
import datetime
from StockList import StockList
from SQL_DB import SQL_DB
from Graph import Graph
from event_logger import Logger

SQL = SQL_DB()
logger = Logger()
start_date = datetime.date(2008,1,1)
end_date = datetime.date(2011,1,1)
stock_list = StockList(SQL, logger, start_date=start_date, end_date=end_date)

key = list(stock_list.stocks.keys())[0]
first_stock = stock_list.stocks[key]

graph = Graph(first_stock, stock_list, logger)
graph.plot()

