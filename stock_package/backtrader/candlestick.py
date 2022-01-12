import datetime
from math import pi
import numpy as np
import pandas as pd
from bokeh.models import HoverTool, ColumnDataSource, CrosshairTool
from bokeh.plotting import figure, show, curdoc
from Stock import Stock


class Graph:
    def __init__(self, df):
        self.df = df


    def plot(self):
        df = self.df

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

        # show(p)
        # layout = column(self.button, fig)
        # show(layout)
        curdoc().add_root(p)

