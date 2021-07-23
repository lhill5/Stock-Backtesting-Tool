from math import pi

import pandas as pd

from bokeh.plotting import figure, show, curdoc
from bokeh.models import HoverTool
from Stock import Stock
import pandas as pd
from math import pi
from bokeh.plotting import figure, show, output_file, curdoc
from bokeh.models import HoverTool, ColumnDataSource, CustomJS, Button, Rect, Select, CrosshairTool
from bokeh.events import ButtonClick, MouseMove, PanStart, PanEnd, Pan, MouseWheel, MouseEnter
from bokeh.layouts import column
from bokeh.palettes import RdYlBu3
from bokeh.themes import built_in_themes
import datetime
import numpy as np


stock = Stock('msft', start_date=datetime.date(2020,1,1), end_date=datetime.date(2021,7,11))
MSFT = stock.stock_dict
df = pd.DataFrame(MSFT)

seqs=np.arange(df.shape[0])
df["seq"]=pd.Series(seqs)
df["date"] = pd.to_datetime(df["date"])

inc = df.close > df.open
dec = df.open > df.close
w=0.3

#use ColumnDataSource to pass in data for tooltips
sourceInc=ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
sourceDec=ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))

#the values for the tooltip come from ColumnDataSource
hover = HoverTool(
    tooltips=[
        ('date', '@date{%F}'),
        ('open', '@open'),
        ('high', '@high'),
        ('low', '@low'),
        ('close', '@close'),
        ('adj_close', '@adj_close'),
        ('volume', '@volume'),
    ],
    formatters={
        '@date': 'datetime',
        'open': 'numeral',
        'high': 'numeral',
        'low': 'numeral',
        'close': 'numeral',
        'adj_close': 'numeral',
        'volume': 'numeral'
    }
)

TOOLS = [CrosshairTool(), hover]
p = figure(plot_width=700, plot_height=400, tools=TOOLS)
p.xaxis.major_label_orientation = pi/4
p.grid.grid_line_alpha=0.3

p.segment('seq', 'high', 'seq', 'low', color="#12C98C", source=sourceInc, name="green_candle")
p.segment('seq', 'high', 'seq', 'low', color="#F2583E", source=sourceDec, name="red_candle")
p.vbar('seq', w, 'open', 'close', fill_color="#12C98C", line_color="#12C98C", source=sourceInc, name="green_candle")
p.vbar('seq', w, 'open', 'close', fill_color="#F2583E", line_color="#F2583E", source=sourceDec, name="red_candle")

show(p)

