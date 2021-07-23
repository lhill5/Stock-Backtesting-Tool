import candlestick as cs
import pandas as pd
from Stock import Stock
import datetime

stock = Stock('msft', start_date=datetime.date(2020,1,1), end_date=datetime.date(2021,7,11))
MSFT = stock.stock_dict
df = pd.DataFrame(MSFT)

cs.plot(df)

