# import pandas as pd
# from pprint import pprint
# from bokeh.palettes import Spectral4
# from bokeh.plotting import figure, show
# from bokeh.sampledata.stocks import AAPL, GOOG, IBM, MSFT
#
# p = figure(plot_width=800, plot_height=250, x_axis_type="datetime")
# p.title.text = 'Click on legend entries to mute the corresponding lines'
#
# for data, name, color in zip([AAPL, IBM, MSFT, GOOG], ["AAPL", "IBM", "MSFT", "GOOG"], Spectral4):
#     df = pd.DataFrame(data)
#     df['date'] = pd.to_datetime(df['date'])
#     p.line(df['date'], df['close'], line_width=2, color=color, alpha=0.8,
#            muted_color="#ff0000", muted_alpha=0.2, legend_label=name)
#
# p.legend.location = "top_left"
# p.legend.click_policy="mute"
# p.legend.inactive_fill_color = "#ff0000"
# p.legend.inactive_fill_alpha = 0.2
#
# pprint(dir(p.legend))
# # p.legend.muted_background_fill_color = "#ff0000"
#
# show(p)
#









import pandas as pd
import datetime
from bokeh.plotting import show
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from stock_package.backtrader.Stock import Stock
from stock_package.backtrader.SQL_DB import SQL_DB

start_date, end_date = datetime.date(2010, 1, 1), datetime.date(2020, 1, 1)
database = SQL_DB(None, update=False)
stock = Stock('aapl', database, start_date=start_date, end_date=end_date)

stock_data = {key: vals for key, vals in stock.stock_dict.items()}
df = pd.DataFrame.from_dict(stock_data)
df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']

# transform date to pandas timestamp (must be this otherwise data doesn't showup on bokeh chart)
df['date'] = pd.to_datetime([x for x in df['date'].squeeze().tolist()], dayfirst=True)
df['str_date'] = [str(d.date()) for d in df['date']]

stock_source = ColumnDataSource(ColumnDataSource.from_df(df))


columns = [
    TableColumn(field='str_date', title='Date'),
    TableColumn(field='open', title='Open'),
    TableColumn(field='high', title='High'),
    TableColumn(field='low', title='Low'),
    TableColumn(field='close', title='Close'),
    TableColumn(field='volume', title='Volume')
]
stock_screener_table = DataTable(source=stock_source, columns=columns, height=800, background="firebrick", css_classes=["test"], auto_edit=True)

show(stock_screener_table)