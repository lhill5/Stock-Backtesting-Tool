import pandas as pd
import datetime
import numpy as np
import pdb
import math
from math import pi, sqrt
from financial_calcs import *
import candlestick_patterns as candlestick_pattern
from Stock import Stock
import trading_strategies as strat
from random_functions import print_df

from bokeh.plotting import figure, show, output_file, curdoc, reset_output
from bokeh.models import tools, HoverTool, CrosshairTool, ColumnDataSource, CustomJS, Button, Rect, Span, CheckboxButtonGroup, DatePicker, MultiChoice, Range1d, LinearAxis, Label, LabelSet, Title, TableColumn, DataTable, HTMLTemplateFormatter, Panel, Tabs, Column, Slider, Spinner
from bokeh.events import ButtonClick, MouseMove, MouseEnter, MouseLeave, MouseWheel, PanStart, PanEnd, Pan
from bokeh.layouts import row, column, gridplot
from bokeh.palettes import RdYlBu3
from bokeh.themes import built_in_themes


class Graph:
    def __init__(self, stock, stocks, ticker_list, database):
        self.ticker = stock.ticker
        self.stock = stock
        self.stocks = stocks
        self.ticker_list = ticker_list
        self.database = database

        self.candlestick_fig = None
        self.EMA_difference_fig = None
        self.MACD_fig = None
        self.RSI_fig = None
        self.minmax_fig = None
        self.buysell_MACD_fig = None

        # intrinsic value metrics
        # _______________________
        # assumptions
        self.BYFCF = 1000
        self.GR = 6
        self.DR = 10
        self.shares_outstanding = 1000
        self.LGR = 3

        self.FCF = []
        self.DF = []

        self.DFCF = []
        self.DPCF = 0

        self.intrinsic_value = 0
        self.intrinsic_value_per_share = 0

        self.init_stock(stock)
        self.calculate_intrinsic_value()


    def init_stock(self, stock=None, ticker=None):
        # must pass either a "Stock" object or ticker name in order to initialize the currently graphed stock
        if stock is None and ticker is None:
            return

        if ticker:
            start_date, end_date = datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)
            stock = Stock(ticker, self.stock.SQL, start_date=start_date, end_date=end_date)

        self.stock = stock
        self.ticker = stock.ticker
        self.stock_dict = stock.stock_dict
        self.tech_indicators = stock.tech_indicators
        self.date_to_index = stock.date_to_index

        self.moving_averages = self.tech_indicators['EMA'].keys()
        self.df = self.get_stock_df()
        self.stocks_df = self.get_stocks_df()

        self.len = self.df.shape[0]
        self.start, self.end = 0, self.len - 1
        self.start_date, self.end_date = convert_str_to_date(self.stock_dict['date'][0]), convert_str_to_date(self.stock_dict['date'][-1])
        self.y_limits = self.get_starting_y_limits()

        _, self.minmax_df = strat.buysell_minmax(stock)
        self.buysell_MACD_df = strat.buysell_MACD1(stock.dates, stock.prices['high'], stock.tech_indicators)

        self.minmax_df['seq'] = self.df['seq']
        self.buysell_MACD_df['seq'] = self.df['seq']

        self.ticker_list_df = self.get_ticker_list_df()
        self.stock_metrics_df = self.get_stock_metrics_df()

        self.stock_source = ColumnDataSource(ColumnDataSource.from_df(self.df))
        self.stocks_source = ColumnDataSource(ColumnDataSource.from_df(self.stocks_df))
        self.minmax_source = ColumnDataSource(ColumnDataSource.from_df(self.minmax_df))
        self.buysell_MACD_source = ColumnDataSource(ColumnDataSource.from_df(self.buysell_MACD_df))

        # self.minmax_rst_source = ColumnDataSource(ColumnDataSource.from_df(self.minmax_rst_df))
        self.ticker_list_source = ColumnDataSource(ColumnDataSource.from_df(self.ticker_list_df))
        self.stock_metrics_source = ColumnDataSource(ColumnDataSource.from_df(self.stock_metrics_df))

        # user for interactivity tools (draw support/resistance lines, draw drag-highlight feature to zoom into plot area, click button to toggle between draw lines / zoom features)
        self.draw_lines_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.draw_rect_source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[]))
        self.button_source = ColumnDataSource(data=dict(value=[True]))

        self.wheel_prev_x = -1

        # creates bokeh figures if they don't already exist
        if self.candlestick_fig is None and self.MACD_fig is None and self.RSI_fig is None:
            TOOLS = 'wheel_zoom, reset, save'
            self.candlestick_fig = figure(title="stock prices", title_location='above', x_axis_type='linear', width=1400, height=400, toolbar_location="right", tools=TOOLS, x_range=(self.start, self.end), y_range=(self.y_limits[0], self.y_limits[1]))
            self.EMA_difference_fig = figure(x_axis_type='linear', tools="crosshair", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range)
            self.above_candle_fig = figure(x_axis_type='linear', tools="crosshair", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range)
            self.candlestick_pattern_fig = figure(x_axis_type='linear', tools="crosshair", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range)

            self.MACD_fig = figure(x_axis_type='linear', tools="crosshair", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range)
            self.RSI_fig = figure(x_axis_type='linear', tools="crosshair", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range, y_range=(0, 100))
            self.minmax_fig = figure(x_axis_type='linear', tools="crosshair", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range)
            self.buysell_MACD_fig = figure(x_axis_type='linear', tools="crosshair", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range)

            # breakpoint()
            crosshair = CrosshairTool(
                dimensions='height',
                line_color="#cbcbcb",
                line_width=0.6
            )
            self.candlestick_fig.add_tools(crosshair)
            self.EMA_difference_fig.add_tools(crosshair)
            self.above_candle_fig.add_tools(crosshair)
            self.MACD_fig.add_tools(crosshair)
            self.RSI_fig.add_tools(crosshair)
            self.minmax_fig.add_tools(crosshair)
            self.buysell_MACD_fig.add_tools(crosshair)

            # initializes event listeners for interactivity tools
            self.init_events()
            self.init_button()
            self.init_autoscale()
        else:
            self.candlestick_fig.x_range.start = self.start
            self.candlestick_fig.x_range.end = self.end
            self.candlestick_fig.y_range.start = self.y_limits[0]
            self.candlestick_fig.y_range.start = self.y_limits[1]
            self.MACD_fig.x_range = self.candlestick_fig.x_range
            self.RSI_fig.x_range = self.candlestick_fig.x_range
            self.minmax_fig.x_range = self.candlestick_fig.x_range
            self.buysell_MACD_fig.x_range = self.candlestick_fig.x_range

    def plot(self):
        # ____ hover tooltip ____
        # hover = HoverTool(
        #     names=[
        #         "green_candle",
        #         "red_candle"
        #     ],
        #     tooltips=[
        #         ('date', '@date{%F}'),
        #         ('open', '@open'),
        #         ('high', '@high'),
        #         ('low', '@low'),
        #         ('close', '@close'),
        #         ('adj_close', '@adj_close'),
        #         ('volume', '@volume'),
        #     ],
        #     formatters={
        #         '@date': 'datetime',
        #         'open': 'numeral',
        #         'high': 'numeral',
        #         'low': 'numeral',
        #         'close': 'numeral',
        #         'adj_close': 'numeral',
        #         'volume': 'numeral'
        #     },
        #     mode='vline'
        # )
        # if replotting this graph, this will remove previous stock/start-end range hover tool, and the updated new hover tool
        # self.remove_prev_hover_tool()
        # self.candlestick_fig.add_tools(hover)

        # records all glyphs to be modified/changed later (used in change_stock function when plotting new stock data)

        self.glyphs = []
        self.tech_indicator_plots = []

        # creates highlight box drawn by user
        rect = Rect(x='x', y='y', width='width', height='height',
                    fill_alpha=0.3, fill_color='#009933')
        rect_glyph = self.candlestick_fig.add_glyph(self.draw_rect_source, rect, selection_glyph=rect, nonselection_glyph=rect)
        self.glyphs.append(rect_glyph)

        # creates lines drawn by user (can be used to draw support/resistance levels)
        line_glyph = self.candlestick_fig.multi_line('x', 'y', line_width=2, source=self.draw_lines_source)
        self.glyphs.append(line_glyph)

        self.plot_candlestick_graph()
        self.plot_EMA_difference_graph()
        self.plot_EMA_above_candlestick()
        self.plot_candlestick_patterns()
        self.plot_MACD_graph()
        self.plot_RSI_graph()
        self.plot_minmax_graph()
        self.plot_buysell_MACD_graph()

        # creates data-tables
        self.plot_stock_prices_table()
        self.plot_trading_strategy_table()
        self.plot_ticker_list_table()
        self.plot_stock_metrics_table()

        # adds interactive widgets (date slicer, stock picker, etc.)
        self.add_interactive_tools()

        # autoscale axes when first plotting stock (accounts for padding between candlestick chart and volume bars)
        self.autoscale_candlestick_yaxis(1, 1, 1)
        self.autoscale_MACD_yaxis(1, 1, 1)
        self.autoscale_minmax_yaxis(1, 1, 1)
        self.autoscale_buysell_MACD_yaxis(1, 1, 1)

        # breakpoint()
        tab0 = Panel(child=row(self.stock_prices_table, self.trading_strategy_table), title='stock prices')

        tab1 = Panel(child=column(row(self.start_date_slicer, self.end_date_slicer, self.select_stock), self.candlestick_fig, self.MACD_fig, self.buysell_MACD_fig), title='candlestick chart')
        tab2 = Panel(child=column(self.BYFCF_spinner, self.GR_spinner, self.all_stocks_table, self.minmax_fig), title='tabular view')
        tab3 = Panel(child=row(self.stock_prices_table), title='stock prices')
        tab4 = Panel(child=row(self.BYFCF_spinner, self.GR_spinner), title='financial data')

        layout = Tabs(tabs=[tab0, tab1, tab2, tab3, tab4])
        curdoc().add_root(layout)


    def plot_candlestick_graph(self):
        df = self.df

        inc = df.close > df.open
        dec = df.open > df.close
        candle_source_inc = ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
        candle_source_dec = ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))

        # candlestick drawing
        candleline_up = self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color="#12C98C", source=candle_source_inc)
        candleline_down = self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color="#F2583E", source=candle_source_dec)
        # candleline_buy = self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color="#12C98C", source=macd_source_buy)
        # candleline_sell = self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color="#F2583E", source=macd_source_sell)

        w = 0.3
        candle_up = self.candlestick_fig.vbar('seq', w, 'open', 'close', fill_color="#12C98C", line_color="#12C98C", source=candle_source_inc, name="green_candle")
        candle_down = self.candlestick_fig.vbar('seq', w, 'open', 'close', fill_color="#F2583E", line_color="#F2583E", source=candle_source_dec, name="red_candle")
        # candle_buy = self.candlestick_fig.vbar('seq', w, 'open', 'close', fill_color="#12C98C", line_color="#12C98C", source=macd_source_buy, name="buy_candle")
        # candle_sell = self.candlestick_fig.vbar('seq', w, 'open', 'close', fill_color="#F2583E", line_color="#F2583E", source=macd_source_sell, name="buy_candle")

        self.glyphs.extend([candleline_up, candleline_down, candle_up, candle_down])

        # ____ add moving averages to plot ____
        moving_avg_colors = ['orange', 'green', 'purple', 'yellow']
        EMAs = [self.candlestick_fig.line('seq', f'EMA_{avg_num}', source=self.stock_source, color=moving_avg_colors[i], name=f"EMA_{avg_num}", legend_label=f"EMA_{avg_num}") for i, avg_num in enumerate(self.moving_averages)]
        SMAs = [self.candlestick_fig.line('seq', f'SMA_{avg_num}', source=self.stock_source, color=moving_avg_colors[i], name=f"SMA_{avg_num}", legend_label=f"SMA_{avg_num}") for i, avg_num in enumerate(self.moving_averages)]
        self.tech_indicator_plots.extend(EMAs)
        self.tech_indicator_plots.extend(SMAs)
        self.glyphs.extend(EMAs)
        self.glyphs.extend(SMAs)

        # ____ add volume to plot ____
        #____________________________________________________________________________________________________#
        self.candlestick_fig.extra_y_ranges = {"volume_axis": Range1d(start=0, end=max(self.df['volume'])*5)}
        self.candlestick_fig.right = [] # removes previous second y-axes (happens after start/end date is changed or stock is changed and replot(...) is called)

        self.candlestick_fig.add_layout(LinearAxis(y_range_name="volume_axis"), 'right')
        volume_up = self.candlestick_fig.vbar('seq', w-0.1, 0, 'volume', source=candle_source_inc, y_range_name='volume_axis', fill_color = "rgba(18, 201, 140, 0.35)", line_color = "rgba(18, 201, 140, 0.35)")
        volume_down = self.candlestick_fig.vbar('seq', w-0.1, 0, 'volume', source=candle_source_dec, y_range_name='volume_axis', fill_color = "rgba(242, 88, 62, 0.35)", line_color = "rgba(242, 88, 62, 0.35)")
        self.glyphs.extend([volume_up, volume_down])

        # add legend
        self.candlestick_fig.legend.location = "top_left"
        self.candlestick_fig.legend.click_policy = "hide"
        self.candlestick_fig.legend.orientation = "horizontal"

        self.candlestick_fig.legend.background_fill_color = "#494949"
        self.candlestick_fig.legend.background_fill_alpha = 1
        self.candlestick_fig.legend.label_text_color = "white"

        self.candlestick_fig.legend.inactive_fill_color = "#494949"
        self.candlestick_fig.legend.inactive_fill_alpha = 0.75

        self.candlestick_fig.legend.border_line_width = 1.5
        self.candlestick_fig.legend.border_line_color = "white"
        self.candlestick_fig.legend.border_line_alpha = 0.75

        self.candlestick_fig.xaxis.major_label_orientation = pi / 4
        self.candlestick_fig.grid.grid_line_alpha = 0.3
        self.candlestick_fig.xaxis.visible = False
        self.candlestick_fig.yaxis[-1].visible = False


    def plot_EMA_difference_graph(self):
        df = self.df

        EMA_diff_inc = df['EMA9-200_percent'] >= 0
        EMA_diff_dec = df['EMA9-200_percent'] < 0
        EMA_diff_Incsource=ColumnDataSource(ColumnDataSource.from_df(df.loc[EMA_diff_inc]))
        EMA_diff_Decsource = ColumnDataSource(ColumnDataSource.from_df(df.loc[EMA_diff_dec]))

        w = 0.3
        EMA_diff_up = self.EMA_difference_fig.vbar('seq', w, 'EMA9-200_percent', source=EMA_diff_Incsource, fill_color="#12C98C", line_color="#12C98C")
        EMA_diff_down = self.EMA_difference_fig.vbar('seq', w, 'EMA9-200_percent', source=EMA_diff_Decsource, fill_color="#F2583E", line_color="#F2583E")
        self.glyphs.extend([EMA_diff_up, EMA_diff_down])

        self.EMA_difference_fig.xaxis.visible = False


    def plot_EMA_above_candlestick(self):
        df = self.df

        inc = df['candle_above_EMA9'] == 1
        dec = df['candle_above_EMA9'] == -1
        source_inc = ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
        source_dec = ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))

        w = 0.3
        EMA_above_candle = self.above_candle_fig.vbar('seq', w, 'candle_above_EMA9', source=source_inc, fill_color="#12C98C", line_color="#12C98C")
        EMA_below_candle = self.above_candle_fig.vbar('seq', w, 'candle_above_EMA9', source=source_dec, fill_color="#F2583E", line_color="#F2583E")
        self.glyphs.extend([EMA_above_candle, EMA_below_candle])

        self.above_candle_fig.xaxis.visible = False


    def plot_candlestick_patterns(self):
        df = self.df

        inc = df['bullish_3_line_strike'] == 1
        dec = df['bullish_3_line_strike'] == 0
        source_inc = ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
        source_dec = ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))

        w = 0.3
        candlestick_pattern_up = self.candlestick_pattern_fig.vbar('seq', w, 'bullish_3_line_strike', source=source_inc, fill_color="#12C98C", line_color="#12C98C")
        candlestick_pattern_down = self.candlestick_pattern_fig.vbar('seq', w, 'bullish_3_line_strike', source=source_dec, fill_color="#F2583E", line_color="#F2583E")
        self.glyphs.extend([candlestick_pattern_up, candlestick_pattern_down])

        self.candlestick_pattern_fig.xaxis.visible = False


    def plot_MACD_graph(self):
        df = self.df

        macd_buy = pd.DataFrame([df.loc[i] for i in range(1, len(df)) if df.loc[i - 1, 'histogram'] <= 0 and df.loc[i, 'histogram'] > 0])
        macd_sell = pd.DataFrame([df.loc[i] for i in range(1, len(df)) if df.loc[i - 1, 'histogram'] >= 0 and df.loc[i, 'histogram'] < 0])
        sourceMACD_buy = ColumnDataSource(ColumnDataSource.from_df(macd_buy))
        sourceMACD_sell = ColumnDataSource(ColumnDataSource.from_df(macd_sell))

        hist_inc = df.histogram > 0
        hist_dec = df.histogram <= 0
        histogramInc = ColumnDataSource(ColumnDataSource.from_df(df.loc[hist_inc]))
        histogramDec = ColumnDataSource(ColumnDataSource.from_df(df.loc[hist_dec]))

        # add MACD indicators
        w = 0.3
        histogram_up = self.MACD_fig.vbar('seq', w, 'histogram', source=histogramInc, fill_color="#12C98C", line_color="#12C98C")
        histogram_down = self.MACD_fig.vbar('seq', w, 'histogram', source=histogramDec, fill_color="#F2583E", line_color="#F2583E")
        macd = self.MACD_fig.line('seq', 'MACD', source=self.stock_source, line_color='orange')
        signal = self.MACD_fig.line('seq', 'signal', source=self.stock_source, line_color='#5985FF')

        self.tech_indicator_plots.append(self.MACD_fig)
        self.glyphs.extend([histogram_up, histogram_down, macd, signal])

        self.MACD_fig.xaxis.visible = False


    def plot_RSI_graph(self):

        # add RSI indicator
        rsi = self.RSI_fig.line('seq', 'RSI', source=self.stock_source, line_color='#04BFDC')
        hline_30 = Span(location=30, dimension='width', line_color='red', line_dash='solid', line_width=2)
        hline_70 = Span(location=70, dimension='width', line_color='red', line_dash='solid', line_width=2)
        self.RSI_fig.add_layout(hline_30)
        self.RSI_fig.add_layout(hline_70)

        self.tech_indicator_plots.append(self.RSI_fig)
        self.glyphs.append(rsi)

        # text box to show which date is currently being hovered over on chart
        self.date_hover_label = Label(x=83, y=0, x_units='data', y_units='data',
                                      text='Date: 2021-02-07', text_color="white", text_align='center', text_font_size='16px', render_mode='css',
                                      background_fill_color='#d3d3d3', background_fill_alpha=0.4)
        self.RSI_fig.add_layout(self.date_hover_label)

        self.RSI_fig.xaxis.visible = False


    def plot_minmax_graph(self):
        minmax = self.minmax_fig.line('seq', 'high', source=self.stock_source, line_color='#ffa500')
        self.minmax_fig.scatter('seq', 'buy', marker="circle", source=self.minmax_source, color="green")
        self.minmax_fig.scatter('seq', 'sell', marker="circle", source=self.minmax_source, color="red")

        self.tech_indicator_plots.append(self.minmax_fig)
        self.glyphs.append(minmax)

        # text box to show which date is currently being hovered over on chart
        # self.date_hover_label = Label(x=83, y=0, x_units='data', y_units='data',
        #                               text='Date: 2021-02-07', text_color="white", text_align='center', text_font_size='16px', render_mode='css',
        #                               background_fill_color='#d3d3d3', background_fill_alpha=0.4)
        # self.minmax_fig.add_layout(self.date_hover_label)

        self.minmax_fig.xaxis.visible = False


    def plot_buysell_MACD_graph(self):
        buysell_MACD = self.buysell_MACD_fig.line('seq', 'high', source=self.stock_source, line_color='#ffa500')
        self.buysell_MACD_fig.scatter('seq', 'buy', marker="circle", source=self.buysell_MACD_source, color="green")
        self.buysell_MACD_fig.scatter('seq', 'sell', marker="circle", source=self.buysell_MACD_source, color="red")

        self.tech_indicator_plots.append(self.buysell_MACD_fig)
        self.glyphs.append(buysell_MACD)

        # text box to show which date is currently being hovered over on chart
        # self.date_hover_label = Label(x=83, y=0, x_units='data', y_units='data',
        #                               text='Date: 2021-02-07', text_color="white", text_align='center', text_font_size='16px', render_mode='css',
        #                               background_fill_color='#d3d3d3', background_fill_alpha=0.4)
        # self.buysell_MACD_fig.add_layout(self.date_hover_label)

        self.buysell_MACD_fig.xaxis.visible = False


    def plot_stock_prices_table(self):
        columns = [
            TableColumn(field='str_date', title='Date'),
            TableColumn(field='open', title='Open'),
            TableColumn(field='high', title='High'),
            TableColumn(field='low', title='Low'),
            TableColumn(field='close', title='Close'),
            TableColumn(field='adj_close', title='Adj. Close'),
            TableColumn(field='volume', title='Volume'),
            TableColumn(field='percent_change_str', title='% Change', formatter=self.get_html_formatter())
        ]
        self.stock_prices_table = DataTable(source=self.stock_source, columns=columns, height=800, css_classes=["table_rows"], index_position=None, margin=(5,5,5,25))


    def plot_trading_strategy_table(self):

        columns = [
            TableColumn(field='ticker', title='Ticker'),
            # TableColumn(field='total transactions', title='total transactions'),
            # TableColumn(field='buy/sell profit', title='buy/sell profit'),
            # TableColumn(field='total profit', title='total profit'),
            # TableColumn(field='buy/sell gain', title='buy/sell gain'),
            TableColumn(field='minmax buy/sell CAGR', title='buy/sell CAGR'),
            TableColumn(field='minmax CAGR', title='minmax CAGR'),
            TableColumn(field='MACD1 CAGR', title='MACD CAGR')
        ]
        self.trading_strategy_table = DataTable(source=self.stocks_source, columns=columns, height=800, autosize_mode='fit_viewport', css_classes=["table_rows"], index_position=None, margin=(5,5,5,25))


    def plot_ticker_list_table(self):
        columns = [
            TableColumn(field='stocks', title='Stocks'),
            TableColumn(field='intrinsic_value', title='Intrinsic Value')
        ]
        self.all_stocks_table = DataTable(source=self.ticker_list_source, columns=columns, height=800, index_position=None, margin=(5, 5, 5, 25))


    def plot_stock_metrics_table(self):
        columns = [
            TableColumn(field='stock', title='Stock'),
            TableColumn(field='intrinsic_value', title='Intrinsic Value'),
            TableColumn(field='intrinsic_value_per_share', title='Intrinsic Value Per Share')
        ]
        self.stock_metrics_table = DataTable(source=self.stock_metrics_source, columns=columns, height=800, index_position=None, margin=(5, 5, 5, 25), editable=True)


    def plot_stock_screener_table(self):
        pass


    def add_interactive_tools(self):
        # date slicer to select start/end date
        start_date_slicer = DatePicker(title='Start Time', value=self.start_date, min_date=self.stock.valid_start_date_offset, max_date=self.end_date, width=115, height=50, margin=(5, 5, 5, 880))
        start_date_slicer.on_change("value", self.start_date_change)
        self.start_date_slicer = start_date_slicer

        end_date_slicer = DatePicker(title='End Time', value=self.end_date, min_date=self.start_date, max_date=self.stock.valid_end_date, width=115, height=50, margin=(5, 5, 5, 20))
        end_date_slicer.on_change("value", self.end_date_change)
        self.end_date_slicer = end_date_slicer

        # toggle which buttons are active initially
        self.button_labels = ['EMA 9', 'EMA 50', 'EMA 150', 'EMA 200', 'SMA 9', 'SMA 50', 'SMA 150', 'SMA 200', 'MACD', 'RSI']
        A = list(range(len(self.tech_indicator_plots)))
        B = [2, 4, 5, 6, 7]
        default_active_plots = list(set(A) - set(B))

        # default_active_plots = [0, 1, 3, 8, 9, 10]  # add 9 for RSI
        for i, tech_ind in enumerate(self.tech_indicator_plots):
            tech_ind.visible = i in default_active_plots

        # buttons to select technical indicators to show/hide
        # self.checkbox_button_group = CheckboxButtonGroup(labels=self.button_labels, active=default_active_plots, button_type="primary", css_classes=["button_margin"], height=30, margin=(23, 5, 5, 20))
        # self.checkbox_button_group.on_click(self.click_tech_indicators_button)

        # search bar for selecting which stock to view
        OPTIONS = self.ticker_list
        self.select_stock = MultiChoice(value=[self.ticker.upper()], options=OPTIONS, css_classes=["stock_picker_color"], width=210, max_items=1, margin=(17, 15, 5, 20))
        self.select_stock.on_change("value", self.stock_change)

        self.BYFCF_spinner = Spinner(low=0, high=1000000000, value=1000, step=1000, width=120, title="Base Year Free Cash Flow (M)")
        self.BYFCF_spinner.on_change("value", self.update_BYFCF)

        self.GR_spinner = Spinner(low=0, high=25, value=5, step=.1, width=60, margin=(5,5,5,60), title="Company's Growth Rate (%)")
        self.GR_spinner.on_change("value", self.update_GR)


    def draw_line_event(self):
        return CustomJS(args=dict(source=self.draw_lines_source, button_source=self.button_source), code="""
            const data = source.data;
            const button_value = button_source.data['value'][0]
                        
            if (!button_value) {
    
                let x = cb_obj['x'];
                let y = cb_obj['y'];
                let event = cb_obj.event_name;
    
                if (event === 'panstart') {
                    localStorage.setItem('x', x);
                    localStorage.setItem('y', y);
                    localStorage.setItem('num_points', data['x'].length);
                }
                else if (event === 'panend' || event === 'pan') {
                    let x0 = localStorage.getItem('x');
                    let y0 = localStorage.getItem('y');
                    let x1 = x;
                    let y1 = y;
    
                    // check to see if we should add point or modify previous point if not the first pan event for this line
                    let num_points = localStorage.getItem('num_points');
                    let arr_len = data['x'].length;
                    // check if this is the first pan event, meaning we need to add the point instead of modify previous point
                    if (event === 'pan' && num_points == arr_len) {
                        data['x'].push([x0, x1]);
                        data['y'].push([y0, y1]);
                    }
                    // modify previous point (allows user to move line around while dragging mouse)
                    else {
                        data['x'][arr_len-1][1] = x1;
                        data['y'][arr_len-1][1] = y1;
                    }
    
                    source.change.emit();
                }
            }
        """)


    def draw_rect_event(self):
        return CustomJS(args=dict(source=self.draw_rect_source, button_source=self.button_source, x_range=self.candlestick_fig.x_range, y_limits=(self.y_limits[0], self.y_limits[1])), code="""
            const data = source.data
            
            const y_min = y_limits[0]
            const y_max = y_limits[1]
            const button_value = button_source.data['value'][0] 
            
            if (button_value) {
    
                let x, y, width, height
                height = y_max - y_min
                y = y_min + height/2
    
                let event = cb_obj.event_name
                
                if (event == 'panstart') {
                    width = 0
                    x = cb_obj['x'] + (width / 2)
                    localStorage.setItem('line_x0', x)
                }
                else if (event == 'pan') {
                    let x0 = Number(localStorage.getItem('line_x0'))
                    width = cb_obj['x'] - x0
                    x = x0 + (width / 2)
                    localStorage.setItem('line_x1', x0 + width)
                }
    
                data['x'].pop()
                data['y'].pop()
                data['width'].pop()
                data['height'].pop()
    
                // removes rectangle once user release the mouse-click
                if (event !== 'panend') {
                    data['x'].push(x)
                    data['y'].push(y)
                    data['width'].push(width)
                    data['height'].push(height)
                }
                
                // updates x_axis start/end values based on which direction user scrolled
                if (event === 'panend') {
                    let x0 = Number(localStorage.getItem('line_x0'))
                    let x1 = Number(localStorage.getItem('line_x1'))
                    
                    // swap if start is greater than end (start should always be before end)
                    if (x0 > x1) {
                        let tmp = x0
                        x0 = x1
                        x1 = tmp
                    }
                    
                    x_range.start = Math.max(Math.floor(x0), x_range.start)
                    x_range.end = Math.min(Math.ceil(x1), x_range.end)
                }
                
                // emit update of data source
                source.change.emit()
            }
        """)


    def show_hover_prices(self, cb_obj):
        x = int(round(cb_obj.x, 0))
        in_range = x >= 0 and x < len(self.df)

        if cb_obj.event_name == 'mouseleave' or not in_range:
            self.date_hover_label.text = ""
            self.candlestick_fig.title.text = ""

        if not in_range:
            return

        date = self.df.loc[x, 'date'].date()
        open_price = self.df.loc[x, 'open']
        high_price = self.df.loc[x, 'high']
        low_price = self.df.loc[x, 'low']
        close_price = self.df.loc[x, 'close']
        percent_change = self.df.loc[x, 'percent_change']

        if cb_obj.event_name == 'mouseenter' or cb_obj.event_name == 'mousemove':
            # change date label to move with hovering cursor
            self.date_hover_label.text = f'{date}'
            self.date_hover_label.x = x

            title = f"O: {open_price}   H: {high_price}   L: {low_price}   C: {close_price} ({percent_change}%)"
            self.candlestick_fig.title.text = title
            if percent_change >= 0:
                self.candlestick_fig.title.text_color = 'green'
            else:
                self.candlestick_fig.title.text_color = 'red'


    def move_chart_event(self, cb_obj):

        x_range = self.candlestick_fig.x_range
        delta = 1 if cb_obj.delta > 0 else -1

        if self.wheel_prev_x == -1:
            self.wheel_prev_x = cb_obj.sx
            # print('wheel x is -1')
            return

        move_speed = round((0.0175 * (x_range.end - x_range.start)), 1)
        # print(f'x start: {round(x_range.start,1)}, x end: {round(x_range.end,1)}, move speed: {move_speed}')
        # move_speed = 3
        if x_range.start - (move_speed * delta) >= self.start and x_range.end - (move_speed * delta) <= self.end:
            x_range.start -= move_speed * delta
            x_range.end -= move_speed * delta


    # def click_tech_indicators_button(self, active_plots):
    #     # pdb.set_trace()
    #     for i, tech_ind in enumerate(self.tech_indicator_plots):
    #         tech_ind.visible = i in active_plots


    def start_date_change(self, attr, old_date, new_date):
        # if new start date is not within valid range of dates, then recreate stock obj with new start/end params.
        # pdb.set_trace()
        new_date = convert_str_to_date(new_date)
        if not (new_date >= self.start_date and new_date <= self.end_date):
            stock = Stock(self.stock.ticker, self.stock.SQL, start_date=new_date, end_date=self.end_date)
            self.init_stock(stock=stock)
            self.reset_charts()
            self.replot()

        date_index = self.date_to_index[new_date]
        x_range = self.candlestick_fig.x_range
        x_range.start = date_index


    def end_date_change(self, attr, old_date, new_date):
        # if new end date is not within valid range of dates, then recreate stock obj with new start/end params.
        new_date = convert_str_to_date(new_date)
        if not (new_date >= self.start_date and new_date <= self.end_date):
            stock = Stock(self.stock.ticker, self.stock.SQL, start_date=self.start_date, end_date=new_date)
            self.init_stock(stock=stock)
            self.reset_charts()
            self.replot()

        date_index = self.date_to_index[new_date]
        x_range = self.candlestick_fig.x_range
        x_range.end = date_index


    def stock_change(self, attr, old_ticker, new_ticker):
        # if len(new_ticker) != 0:
        #     pdb.set_trace()

        if new_ticker is not None and len(new_ticker) != 0:
            self.init_stock(ticker=new_ticker[-1])
            self.reset_charts()
            self.replot()


    def update_BYFCF(self, attr, old, new):
        self.BYFCF = round(new, 2)
        self.calculate_intrinsic_value()


    def update_GR(self, attr, old, new):
        self.GR = round(new, 2)
        self.calculate_intrinsic_value()


    # calculates the intrinsic value of a company based on the "Discount Cash Flow (DCF) Intrinsic Value Model"
    def calculate_intrinsic_value(self):
        # calculate free cash flow for next 10 years
        self.FCF = [self.BYFCF * pow(1+(self.GR/100), n) for n in range(1, 11)]
        # print(self.FCF)
        # calculate discount factor for next 10 years
        self.DF = [pow(1+(self.DR/100), n) for n in range(1, 11)]
        # print(self.DF)
        # calculate discounted free cash flow from above calculations
        self.DFCF = [FCF/DF for FCF, DF in zip(self.FCF, self.DF)]
        # print(self.DFCF)
        # calculate discounted perpetuity free cash flow beyond 10 years
        self.DPCF = divide((self.BYFCF * pow((1+(self.GR/100)), 11) * (1+(self.LGR/100))), ((self.DR - self.LGR)/100)) * (divide(1, pow(1+(self.DR/100), 11)))
        # print(self.DPCF)
        # finally calculate the intrinsic value based on above calculations
        self.intrinsic_value = round(sum(self.DFCF) + self.DPCF, 2)
        # print(self.intrinsic_value)
        # another useful metric for showing the intrinsic value per share and comparing this with current share price ($)
        self.intrinsic_value_per_share = round(divide(self.intrinsic_value, self.shares_outstanding), 2)
        self.update_stock_metrics()


    def update_stock_metrics(self):
        self.stock_metrics_source.data['intrinsic_value'] = [self.intrinsic_value]
        self.stock_metrics_source.data['intrinsic_value_per_share'] = [self.intrinsic_value_per_share]


    def init_events(self):
        point_events = [Pan, PanStart, PanEnd]

        events = [Pan, PanStart, PanEnd]
        for event in events:
            self.candlestick_fig.js_on_event(event, self.draw_rect_event())
            self.candlestick_fig.js_on_event(event, self.draw_line_event())

        events = [MouseMove, MouseEnter, MouseLeave]
        for event in events:
            self.candlestick_fig.on_event(event, self.show_hover_prices)

        self.candlestick_fig.on_event(MouseWheel, self.move_chart_event)


    def button_callback(self):
        pass


    def init_button(self):
        self.button = Button(label="GFG")
        self.button.js_on_click(
            CustomJS(args=dict(source=self.button_source), code="""
                const data = source.data

                data['value'][0] = !data['value'][0]

                console.log(data['value'][0])
                source.change.emit()            
            """)
        )


    def init_autoscale(self):
        self.candlestick_fig.x_range.on_change('start', self.autoscale_candlestick_yaxis)
        self.candlestick_fig.x_range.on_change('end', self.autoscale_candlestick_yaxis)
        self.MACD_fig.x_range.on_change('start', self.autoscale_MACD_yaxis)
        self.MACD_fig.x_range.on_change('end', self.autoscale_MACD_yaxis)
        self.minmax_fig.x_range.on_change('start', self.autoscale_minmax_yaxis)
        self.minmax_fig.x_range.on_change('end', self.autoscale_minmax_yaxis)
        self.buysell_MACD_fig.x_range.on_change('start', self.autoscale_buysell_MACD_yaxis)
        self.buysell_MACD_fig.x_range.on_change('end', self.autoscale_buysell_MACD_yaxis)


    def autoscale_candlestick_yaxis(self, attr, old, new):
        # source = ColumnDataSource({'date': self.df.date, 'high': self.df.high, 'low': self.df.low, 'index': [i for i in range(len(self.df.date))]})
        # pdb.set_trace()
        index = [i for i in range(len(self.df.date))]
        high = self.df.high
        low = self.df.low
        volume = self.df.volume
        # active_tech_indicators = [self.tech_indicator_plots[i].name for i in self.checkbox_button_group.active]
        # tech_ind_data = [self.df[ind] for ind in active_tech_indicators if ind is not None]
        active_tech_indicators = [self.tech_indicator_plots[i].name for i in range(len(self.tech_indicator_plots)) if self.tech_indicator_plots[i].visible]
        tech_ind_data = [self.df[ind] for ind in active_tech_indicators if ind is not None]

        x_range = self.candlestick_fig.x_range
        y_range = self.candlestick_fig.y_range
        start, end = x_range.start, x_range.end

        min_val = math.inf
        max_val = -math.inf
        max_volume = -math.inf

        for i in index:
            if i >= start and i <= end:
                max_val = max(high[i], max([ind_val[i] for ind_val in tech_ind_data], default=-math.inf), max_val)
                min_val = min(low[i], min([ind_val[i] for ind_val in tech_ind_data], default=math.inf), min_val)
                max_volume = max(volume[i], max_volume)
        pad = (max_val - min_val) * 0.05

        # pdb.set_trace()
        volume_range = self.candlestick_fig.extra_y_ranges['volume_axis'].end - self.candlestick_fig.extra_y_ranges['volume_axis'].start
        candlestick_range = self.candlestick_fig.y_range.end - self.candlestick_fig.y_range.start

        # figures out what % the max volume bar takes up on the chart, then pads the bottom by the same % so that candlestick lines/bars don't intersect with volume bars
        max_vol_percent = divide(max_volume, volume_range)
        bottom_padding = max_vol_percent * candlestick_range

        y_range.start = min_val - bottom_padding - (pad/2)
        y_range.end = max_val + pad


    def autoscale_MACD_yaxis(self, attr, old, new):
        index = [i for i in range(len(self.df.date))]
        macd = self.df.MACD
        signal = self.df.signal

        x_range = self.MACD_fig.x_range
        y_range = self.MACD_fig.y_range
        start, end = x_range.start, x_range.end

        min_val = math.inf
        max_val = -math.inf

        for i in index:
            if i >= start and i <= end:
                max_val = max(macd[i], signal[i], max_val)
                min_val = min(macd[i], signal[i], min_val)

        pad = (max_val - min_val) * 0.05

        y_range.start = min_val - pad
        y_range.end = max_val + pad


    def autoscale_minmax_yaxis(self, attr, old, new):
        index = [i for i in range(len(self.df.date))]
        prices = self.df.high # high prices

        x_range = self.minmax_fig.x_range
        y_range = self.minmax_fig.y_range
        start, end = x_range.start, x_range.end

        min_val = math.inf
        max_val = -math.inf

        for i in index:
            if i >= start and i <= end:
                max_val = max(prices[i], max_val)
                min_val = min(prices[i], min_val)

        pad = (max_val - min_val) * 0.05

        y_range.start = min_val - pad
        y_range.end = max_val + pad


    def autoscale_buysell_MACD_yaxis(self, attr, old, new):
        index = [i for i in range(len(self.df.date))]
        prices = self.df.high # high prices

        x_range = self.buysell_MACD_fig.x_range
        y_range = self.buysell_MACD_fig.y_range
        start, end = x_range.start, x_range.end

        min_val = math.inf
        max_val = -math.inf

        for i in index:
            if i >= start and i <= end:
                max_val = max(prices[i], max_val)
                min_val = min(prices[i], min_val)

        pad = (max_val - min_val) * 0.05

        y_range.start = min_val - pad
        y_range.end = max_val + pad


    def get_starting_y_limits(self):
        start_i = self.start
        end_i = self.end

        # start_i = self.date_to_index[start_x]
        # end_i = self.date_to_index[end_x]

        y_min = min(self.stock_dict['low'][start_i:end_i+1])
        y_max = max(self.stock_dict['high'][start_i:end_i+1])

        # 5 percent offset on upper/lower limit of graph
        offset = 0.05
        return y_min - offset*y_min, y_max + offset*y_max


    # coverts stock data into a pandas dataframe
    def get_stock_df(self):
        stock_data = {key: vals for key, vals in self.stock_dict.items()}
        df = pd.DataFrame.from_dict(stock_data)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']

        len = df.shape[0]
        seqs = np.arange(len)
        df["seq"] = pd.Series(seqs)
        # df["date"] = pd.to_datetime(df["date"])

        # transform date to pandas timestamp (must be this otherwise data doesn't showup on bokeh chart)
        df["date"] = pd.to_datetime(df["date"])
        df['str_date'] = [str(d.date()) for d in df['date']]

        # add new columns (technical indicators)
        for indicator, values in self.tech_indicators.items():
            if 'EMA' in indicator or 'SMA' in indicator:
                for num_days, val in values.items():
                    df[f'{indicator}_{num_days}'] = val
            else:
                df[indicator] = values

        # creates percent_change (close - prev_close) / prev_close column, where first % change is 0
        df['percent_change'] = [0] + [round(divide(df.loc[i,'close'] - df.loc[i-1,'close'], df.loc[i-1,'close']) * 100, 2) for i in range(1, len)]
        df['percent_change_str'] = [str(change) + '%' for change in df['percent_change']]
        df['percent_change_color'] = ["#12C98C" if change >= 0 else "#F2583E" for change in df['percent_change']]

        df['green_candle'] = [1 if df.loc[i, 'close'] >= df.loc[i, 'open'] else 0 for i in range(len)]
        df['red_candle'] = [1 if df.loc[i, 'close'] < df.loc[i, 'open'] else 0 for i in range(len)]

        df['EMA9-200_percent'] = [divide((df.loc[i, 'EMA_9'] - df.loc[i, 'EMA_200']), df.loc[i, 'EMA_200'])*100 for i in range(len)]
        df['candle_above_EMA9'] = [1 if low > EMA_9 else -1 for low, EMA_9 in zip(df['low'], df['EMA_9'])]

        df['bullish_3_line_strike'] = candlestick_pattern.get_3_line_strike(df, bullish=True)
        # print(sum(df['bullish_3_line_strike']))
        df['bearish_3_line_strike'] = candlestick_pattern.get_3_line_strike(df, bearish=True)
        return df


    def get_stocks_df(self):
        stock_data = {key: vals for key, vals in self.stock_dict.items()}
        minmax_stocks_df = pd.DataFrame(columns=['ticker', 'minmax total transactions', 'minmax buy/sell profit', 'minmax total profit', 'minmax buy/sell gain', 'minmax buy/sell CAGR', 'minmax CAGR'], index=list(range(0, len(self.stocks))))
        MACD1_stocks_df = pd.DataFrame(columns=['ticker', 'MACD1 total transactions', 'MACD1 buy/sell profit', 'MACD1 total profit', 'MACD1 buy/sell gain', 'MACD1 buy/sell CAGR', 'MACD1 CAGR'], index=list(range(0, len(self.stocks))))

        # breakpoint()

        for i, ticker in enumerate(self.stocks):
            stock = self.stocks[ticker]

            df_minmax_optimal, df_minmax_real = strat.buysell_minmax(stock)
            df_MACD1 = strat.buysell_MACD1(stock.dates, stock.prices['high'], stock.tech_indicators)

            minmax_rst = list(strat.evaluate_strategy(stock, df_minmax_real))

            minmax_rst = [round(val, 2) for val in minmax_rst]
            MACD1_rst = list(strat.evaluate_strategy(stock, df_MACD1))
            try:
                MACD1_rst = [round(val, 2) for val in MACD1_rst]
            except:
                breakpoint()

            minmax_rst.insert(0, ticker.upper())
            MACD1_rst.insert(0, ticker.upper())

            minmax_stocks_df.loc[i, :] = list(minmax_rst)
            MACD1_stocks_df.loc[i, :] = list(MACD1_rst)

        buysell_indicators_df = pd.merge(minmax_stocks_df, MACD1_stocks_df, on="ticker", how="inner")
        return buysell_indicators_df


    def get_ticker_list_df(self):
        stocks_data = {'stocks': self.ticker_list}
        df = pd.DataFrame.from_dict(stocks_data)
        df['intrinsic_value'] = [0 for i in range(len(self.ticker_list))]
        return df


    # def get_minmax_rst_df(self):
    #     table_df = pd.DataFrame({'ticker': [], 'total transactions': [], 'buy/sell profit': [], 'total profit': [], 'buy/sell gain': [], 'total gain': [], 'buy/sell annual gain': [], 'annual gain': []})
    #     # for ticker in self.ticker_list:
    #     #     stock = Stock(ticker, self.database, start_date=self.start_date, end_date=self.end_date)
    #     #     if len(stock.dates) == 0:
    #     #         continue
    #
    #         df, df_trades = strat.buysell_minmax(stock)
    #         total_transactions, buy_sell_profit, total_profit, buy_sell_gain, total_gain, buy_sell_annual_gain, annual_gain = strat.evaluate_strategy(stock, df_trades)
    #         table_df.loc[len(table_df.index)] = [ticker, total_transactions, buy_sell_profit, total_profit, buy_sell_gain, total_gain, buy_sell_annual_gain, annual_gain]
    #     return table_df


    def get_stock_metrics_df(self):
        stock_data = {'stock': [self.ticker.upper()]}
        df = pd.DataFrame.from_dict(stock_data)
        df['intrinsic_value'] = [self.intrinsic_value]
        df['intrinsic_value_per_share'] = [self.intrinsic_value_per_share]
        return df


    def get_tech_ind_labels(self):
        tech_indicator_labels = []
        for MA_num in self.tech_indicators['EMA'].keys():
            tech_indicator_labels.append(f'EMA_{MA_num}')
            tech_indicator_labels.append(f'SMA_{MA_num}')

        tech_indicator_labels.append('MACD')
        tech_indicator_labels.append('RSI')
        return tech_indicator_labels


    # remove all previous glyphs (line charts / bar charts / etc.)
    def reset_charts(self):
        for glyph in self.glyphs:
            glyph.visible = False
        curdoc().clear()

        # plot new stock data


    def replot(self):
        self.plot()
        self.autoscale_candlestick_yaxis(1, 1, 1)
        self.autoscale_MACD_yaxis(1, 1, 1)
        self.autoscale_minmax_yaxis(1, 1, 1)


    def remove_prev_hover_tool(self):
        for tool in self.candlestick_fig.toolbar.tools:
            if isinstance(tool, tools.HoverTool):
                self.candlestick_fig.toolbar.tools.remove(tool)


    def get_html_formatter(self):

        template = """
        <div style="color:<%=percent_change_color%>";>
        <%= value %></div>
        """
        return HTMLTemplateFormatter(template=template)


if __name__ == '__main__':
    pass

