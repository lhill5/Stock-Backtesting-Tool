import time
import gspread
from math import pi
from oauth2client.service_account import ServiceAccountCredentials

from bokeh.plotting import figure, show, output_file, curdoc, reset_output
from bokeh.models import tools, HoverTool, CrosshairTool, ColumnDataSource, CustomJS, Button, Rect, Span, CheckboxButtonGroup, DatePicker, MultiChoice, Range1d, LinearAxis, Label, LabelSet, Title, TableColumn, DataTable, HTMLTemplateFormatter, Panel, Tabs, Column, Slider, Spinner, Div, BoxAnnotation
from bokeh.events import ButtonClick, MouseMove, MouseEnter, MouseLeave, MouseWheel, PanStart, PanEnd, Pan, MenuItemClick, Press, Tap

from global_functions import *
from financial_calcs import *
from Graph_Data import GraphData
from PlotTools import BokehTab
from Stock import Stock


class Graph:
    def __init__(self, stock, stock_list, logger):
        self.ticker = stock.ticker
        self.stock = stock
        self.stock_list = stock_list.stocks
        self.ticker_list = stock_list.tickers
        self.logger = logger

        # list of trading strategies, add to this list whenever you write a new strategy
        # todo - add strat
        self.trading_strategies = ['hold', 'minmax', 'MACD1', 'EMA1', 'EMA2', 'EMA3', 'RSI', 'ADX', 'ADX2', 'test']

        # records all glyphs to be modified/changed later (used in change_stock function when plotting new stock data)
        self.glyphs = []
        self.tech_indicator_plots = []
        self.main_chart_buysell_lines = []
        self.profit_loss_ranges = []
        self.buysell_scatter_charts = []

        # self.buy_color = "#4FA6D8" # blue
        # self.buy_color = "#12C98C" # green
        self.buy_color = "#4FA6D8"  # purple

        # self.sell_color = "#9B88F6" # purple
        self.sell_color = "#F2583E"  # red

        if len(self.stock.dates) == 0:
            self.logger.info(f'cannot plot {self.ticker} stock, no data')
            return

        # -- add strat --

        self.is_replot = False
        self.figures = []

        self.tabs_layout = None
        self.prev_tab = None

        # self.title = Div(text="<h1 style='color:blue;'>testing</h1>", margin=(5, 5, 5, 25))
        # self.init_google_sheets_API()
        self.init_stock(stock)

    def init_stock(self, stock=None, ticker=None):
        # must pass either a "Stock" object or ticker name in order to initialize the currently graphed stock
        if stock is None and ticker is None:
            return

        if ticker:
            start_date, end_date = datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)
            stock = Stock(ticker, self.stock.SQL, self.logger, request_start_date=start_date, request_end_date=end_date)

        # graph data for current stock
        self.data = GraphData(self.stock, self.ticker_list, self.trading_strategies)

        # used in mouse wheel event to scroll range of price data
        self.wheel_prev_x = -1

        # creates bokeh figures if they don't already exist
        if not self.is_replot:
            self.is_replot = True

            TOOLS = 'wheel_zoom, reset, save'
            self.candlestick_fig = figure(title="stock prices", title_location='above', x_axis_type='linear', width=1400, height=400, toolbar_location="right", tools=TOOLS, x_range=(self.data.start, self.data.end), y_range=(self.data.y_limits[0], self.data.y_limits[1]))
            self.EMA_difference_fig = figure(x_axis_type='linear', tools="crosshair", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range)
            self.above_candle_fig = figure(x_axis_type='linear', tools="crosshair", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range)
            self.candlestick_pattern_fig = figure(x_axis_type='linear', tools="crosshair", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range)
            self.figures.extend([self.candlestick_fig, self.EMA_difference_fig, self.above_candle_fig, self.candlestick_pattern_fig])

            # breakpoint()
            crosshair = CrosshairTool(
                dimensions='height',
                line_color="#cbcbcb",
                line_width=0.6
            )

            for fig in self.figures:
                fig.add_tools(crosshair)

        else:
            self.candlestick_fig.x_range.start = self.data.start
            self.candlestick_fig.x_range.end = self.data.end
            # self.candlestick_fig.y_range.start = self.y_limits[0]
            # self.candlestick_fig.y_range.end = self.y_limits[1]

            # sets all other figures to have same x-axis range as candlestick fig (main chart)
            for fig in self.figures:
                if fig != self.candlestick_fig:
                    fig.x_range = self.candlestick_fig.x_range

    def setup_tabs(self):
        # trading_strat_results = Panel(child=row(self.stock_prices_table, self.trading_strategy_table), title='All trading strategy returns')
        #
        # chart tabs
        # hold_tab = BokehTab(self, 'hold', self.start_date_slicer, self.end_date_slicer, self.select_stock)
        # minmax_tab = BokehTab(self, 'minmax', self.start_date_slicer, self.end_date_slicer, self.select_stock)
        # MACD_tab = BokehTab(self, 'MACD1', self.start_date_slicer, self.end_date_slicer, self.select_stock)
        # RSI_tab = BokehTab(self, 'RSI', self.start_date_slicer, self.end_date_slicer, self.select_stock)
        # EMA2_tab = BokehTab(self, 'EMA2', self.start_date_slicer, self.end_date_slicer, self.select_stock)
        # EMA3_tab = BokehTab(self, 'EMA3', self.start_date_slicer, self.end_date_slicer, self.select_stock)
        # ADX_tab = BokehTab(self, 'ADX', self.start_date_slicer, self.end_date_slicer, self.select_stock)
        # ADX2_tab = BokehTab(self, 'ADX2', self.start_date_slicer, self.end_date_slicer, self.select_stock)
        # test_tab = BokehTab(self, 'test', self.start_date_slicer, self.end_date_slicer, self.select_stock)
        #
        # layout = Tabs(tabs=[tab1, tab3, tab5, tab6, tab7])
        # MACD_subplot = SubPlot('MACD', self.stock.MACD, self.plot_MACD_graph, self.candlestick_fig).fig
        # RSI_subplot = SubPlot('RSI', self.stock.RSI, self.plot_RSI_graph, self.candlestick_fig).fig
        # ADX_subplot = SubPlot('ADX', self.stock.ADX, self.plot_ADX_graph, self.candlestick_fig).fig
        #
        # hold_tab.add(self.candlestick_fig, MACD_subplot, self.trading_strategy_table)
        # minmax_tab.add(self.candlestick_fig, MACD_subplot, self.trading_strategy_table)
        # MACD_tab.add(self.candlestick_fig, MACD_subplot)
        # RSI_tab.add(self.candlestick_fig, RSI_subplot)
        # EMA2_tab.add(self.candlestick_fig)
        # EMA3_tab.add(self.candlestick_fig)
        # ADX_tab.add(self.candlestick_fig, ADX_subplot)
        # ADX2_tab.add(self.candlestick_fig, ADX_subplot, RSI_subplot)
        # test_tab.add(self.candlestick_fig, MACD_subplot, self.trading_strategy_table)
        #
        # fundamental tabs
        # fundamental_title = Div(text="All values are in USD Millions.", margin=(5, 5, 5, 25))
        #
        # income_tab = BokehTab(self, 'income_statement', title=fundamental_title)
        # balance_tab = BokehTab(self, 'balance_sheet', title=fundamental_title)
        # cashflow_tab = BokehTab(self, 'cash_flow_statement', title=fundamental_title)
        #
        # income_tab.add(self.income_statement_table, add_widgets=False)
        # balance_tab.add(self.balance_sheet_table, add_widgets=False)
        # cashflow_tab.add(self.cash_flow_table, add_widgets=False)

        trading_strat_tab = BokehTab(self, 'trading results', self.start_date_slicer, self.end_date_slicer, self.select_stock)
        trading_strat_tab.add(self.trading_strategy_table)

    def plot(self):
        rect = Rect(x='x', y='y', width='width', height='height',
                    fill_alpha=0.3, fill_color='#3d85c6')
        rect_glyph = self.candlestick_fig.add_glyph(self.data.draw_rect_source, rect, selection_glyph=rect, nonselection_glyph=rect)
        self.glyphs.append(rect_glyph)

        # creates lines drawn by user (can be used to draw support/resistance levels)
        line_glyph = self.candlestick_fig.multi_line('x', 'y', line_width=2, source=self.data.draw_lines_source)
        self.glyphs.append(line_glyph)

        self.plot_candlestick_graph()
        self.plot_EMA_difference_graph()
        self.plot_EMA_above_candlestick()
        self.plot_candlestick_patterns()

        # creates data-tables
        self.plot_stock_prices_table()
        self.plot_trading_strategy_table()
        self.plot_ticker_list_table()
        self.plot_stock_metrics_table()

        # autoscale axes when first plotting stock (accounts for padding between candlestick chart and volume bars)
        self.autoscale_candlestick_yaxis(1, 1, 1)

        # initializes event listeners for interactivity tools
        self.init_events()
        self.init_button()
        self.init_autoscale()

        # fundamental data tables
        self.plot_income_statement_table()
        self.plot_balance_sheet_table()
        self.plot_cash_flow_table()

        # fundamental charts
        self.plot_income_statement_charts()

        # adds interactive widgets (date slicer, stock picker, etc.)
        self.add_interactive_tools()

        self.setup_tabs()
        curdoc().add_root(BokehTab.tabs_layout)

    def plot_candlestick_graph(self):
        # plots vertical lines where user should buy or sell based on the below indicator
        # self.plot_all_buysell_lines(self.candlestick_fig)
        # self.plot_all_profit_loss_ranges(self.candlestick_fig)

        df = self.data.df
        inc = df.close > df.open
        dec = df.open > df.close
        candle_source_inc = ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
        candle_source_dec = ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))

        # candlestick drawing
        # candleline_up = self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color="#12C98C", source=candle_source_inc)
        # candleline_down = self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color="#F2583E", source=candle_source_dec)
        candleline_up = self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color=self.buy_color, source=candle_source_inc)
        candleline_down = self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color=self.sell_color, source=candle_source_dec)

        w = 0.3
        candle_up = self.candlestick_fig.vbar('seq', w, 'open', 'close', fill_color=self.buy_color, line_color=self.buy_color, source=candle_source_inc, name="green_candle")
        candle_down = self.candlestick_fig.vbar('seq', w, 'open', 'close', fill_color=self.sell_color, line_color=self.sell_color, source=candle_source_dec, name="red_candle")

        self.glyphs.extend([candleline_up, candleline_down, candle_up, candle_down])

        # ____ add moving averages to plot ____
        moving_avg_colors = ["#0abab5", '#eee600', '#ff6347', '#00755e']
        EMAs = [self.candlestick_fig.line('seq', f'EMA_{avg_num}', source=self.data.stock_source, color=moving_avg_colors[i], name=f"EMA_{avg_num}", legend_label=f"EMA_{avg_num}") for i, avg_num in enumerate(self.data.moving_averages)]
        SMAs = [self.candlestick_fig.line('seq', f'SMA_{avg_num}', source=self.data.stock_source, color=moving_avg_colors[i], name=f"SMA_{avg_num}", legend_label=f"SMA_{avg_num}") for i, avg_num in enumerate(self.data.moving_averages)]
        self.tech_indicator_plots.extend(EMAs)
        self.tech_indicator_plots.extend(SMAs)
        self.glyphs.extend(EMAs)
        self.glyphs.extend(SMAs)

        # ____ add volume to plot ____
        # ____________________________________________________________________________________________________#
        self.candlestick_fig.extra_y_ranges = {"volume_axis": Range1d(start=0, end=max(self.data.df['volume']) * 5)}
        self.candlestick_fig.right = []  # removes previous second y-axes (happens after start/end date is changed or stock is changed and replot(...) is called)

        self.candlestick_fig.add_layout(LinearAxis(y_range_name="volume_axis"), 'right')
        # volume_up = self.candlestick_fig.vbar('seq', w-0.1, 0, 'volume', source=candle_source_inc, y_range_name='volume_axis', fill_color = "rgba(18, 201, 140, 0.35)", line_color = "rgba(18, 201, 140, 0.35)")
        # volume_down = self.candlestick_fig.vbar('seq', w-0.1, 0, 'volume', source=candle_source_dec, y_range_name='volume_axis', fill_color = "rgba(242, 88, 62, 0.35)", line_color = "rgba(242, 88, 62, 0.35)")
        volume_up = self.candlestick_fig.vbar('seq', w - 0.1, 0, 'volume', source=candle_source_inc, y_range_name='volume_axis', fill_color=self.buy_color, line_color=self.buy_color)
        volume_down = self.candlestick_fig.vbar('seq', w - 0.1, 0, 'volume', source=candle_source_dec, y_range_name='volume_axis', fill_color=self.sell_color, line_color=self.sell_color)

        self.glyphs.extend([volume_up, volume_down])

        # add legend
        self.candlestick_fig.legend.location = "top_left"
        self.candlestick_fig.legend.click_policy = "hide"
        self.candlestick_fig.legend.orientation = "horizontal"

        self.candlestick_fig.legend.background_fill_color = "#3D3F47"
        self.candlestick_fig.legend.background_fill_alpha = 1
        self.candlestick_fig.legend.label_text_color = "white"

        self.candlestick_fig.legend.inactive_fill_color = "#3D3F47"
        self.candlestick_fig.legend.inactive_fill_alpha = 0.75

        self.candlestick_fig.legend.border_line_width = 1.5
        self.candlestick_fig.legend.border_line_color = "white"
        self.candlestick_fig.legend.border_line_alpha = 0.75

        self.candlestick_fig.xaxis.major_label_orientation = pi / 4
        self.candlestick_fig.grid.grid_line_alpha = 0.3
        self.candlestick_fig.xaxis.visible = False
        self.candlestick_fig.yaxis[-1].visible = False

    def plot_EMA_difference_graph(self):
        df = self.data.df

        EMA_diff_inc = df['EMA9-200_percent'] >= 0
        EMA_diff_dec = df['EMA9-200_percent'] < 0
        EMA_diff_Incsource = ColumnDataSource(ColumnDataSource.from_df(df.loc[EMA_diff_inc]))
        EMA_diff_Decsource = ColumnDataSource(ColumnDataSource.from_df(df.loc[EMA_diff_dec]))

        w = 0.3
        EMA_diff_up = self.EMA_difference_fig.vbar('seq', w, 'EMA9-200_percent', source=EMA_diff_Incsource, fill_color="#12C98C", line_color="#12C98C")
        EMA_diff_down = self.EMA_difference_fig.vbar('seq', w, 'EMA9-200_percent', source=EMA_diff_Decsource, fill_color="#F2583E", line_color="#F2583E")
        self.glyphs.extend([EMA_diff_up, EMA_diff_down])

        self.EMA_difference_fig.xaxis.visible = False

    def plot_EMA_above_candlestick(self):
        df = self.data.df

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
        df = self.data.df

        inc = df['bullish_3_line_strike'] == 1
        dec = df['bullish_3_line_strike'] == 0
        source_inc = ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
        source_dec = ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))

        w = 0.3
        candlestick_pattern_up = self.candlestick_pattern_fig.vbar('seq', w, 'bullish_3_line_strike', source=source_inc, fill_color="#12C98C", line_color="#12C98C")
        candlestick_pattern_down = self.candlestick_pattern_fig.vbar('seq', w, 'bullish_3_line_strike', source=source_dec, fill_color="#F2583E", line_color="#F2583E")
        self.glyphs.extend([candlestick_pattern_up, candlestick_pattern_down])

        self.candlestick_pattern_fig.xaxis.visible = False

    def plot_MACD_graph(self, fig):
        df = self.data.df

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
        histogram_up = fig.vbar('seq', w, 'histogram', source=histogramInc, fill_color="#12C98C", line_color="#12C98C")
        histogram_down = fig.vbar('seq', w, 'histogram', source=histogramDec, fill_color="#F2583E", line_color="#F2583E")
        macd = fig.line('seq', 'MACD', source=self.data.stock_source, line_color='orange')
        signal = fig.line('seq', 'signal', source=self.data.stock_source, line_color='#5985FF')

        self.tech_indicator_plots.append(fig)
        self.glyphs.extend([histogram_up, histogram_down, macd, signal])

        fig.xaxis.visible = False

    def plot_RSI_graph(self, fig):

        # add RSI indicator
        rsi = fig.line('seq', 'RSI', source=self.data.stock_source, line_color='#04BFDC')
        hline_30 = Span(location=30, dimension='width', line_color='red', line_dash='solid', line_width=2)
        hline_70 = Span(location=70, dimension='width', line_color='red', line_dash='solid', line_width=2)
        fig.add_layout(hline_30)
        fig.add_layout(hline_70)

        self.tech_indicator_plots.append(fig)
        self.glyphs.append(rsi)

        # text box to show which date is currently being hovered over on chart
        self.date_hover_label = Label(x=83, y=0, x_units='data', y_units='data',
                                      text='Date: 2021-02-07', text_color="white", text_align='center', text_font_size='16px', render_mode='css',
                                      background_fill_color='#d3d3d3', background_fill_alpha=0.4)
        fig.add_layout(self.date_hover_label)

        fig.xaxis.visible = False

    def plot_ADX_graph(self, fig):

        # add ADX indicator
        ADX = fig.line('seq', 'ADX', source=self.data.stock_source, line_color='#faf0e6')
        pos_DI = fig.line('seq', '+DI', source=self.data.stock_source, line_color='#32cd32')
        neg_DI = fig.line('seq', '-DI', source=self.data.stock_source, line_color='#ff0800')
        # hline_20 = Span(location=20, dimension='height', line_color='silver', line_dash='dotdash', line_width=2)
        hline_20 = Span(location=20, dimension='width', line_color='#6A5ACD', line_dash='dashed', line_width=2)
        fig.add_layout(hline_20)

        self.tech_indicator_plots.append(fig)
        self.glyphs.extend([ADX, pos_DI, neg_DI])

        # text box to show which date is currently being hovered over on chart
        # self.date_hover_label = Label(x=83, y=0, x_units='data', y_units='data',
        #                               text='Date: 2021-02-07', text_color="white", text_align='center', text_font_size='16px', render_mode='css',
        #                               background_fill_color='#d3d3d3', background_fill_alpha=0.4)
        # fig.add_layout(self.date_hover_label)

        fig.xaxis.visible = False

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
        self.stock_prices_table = DataTable(source=self.data.stock_source, columns=columns, height=800, autosize_mode='fit_viewport', css_classes=["table_rows"], index_position=None, margin=(5, 5, 5, 25))

    def plot_trading_strategy_table(self):
        # -- add strat --
        columns = [TableColumn(field=f'{strat} CAGR', title=f'{strat} CAGR') for strat in self.trading_strategies]
        columns = [TableColumn(field='ticker', title='Ticker')] + columns

        # breakpoint()
        self.trading_strategy_table = DataTable(source=self.data.buysell_results_source, columns=columns, height=800, autosize_mode='fit_viewport', css_classes=["table_rows"], index_position=None, margin=(5, 5, 5, 25))

    def plot_ticker_list_table(self):
        columns = [
            TableColumn(field='stocks', title='Stocks'),
            TableColumn(field='intrinsic_value', title='Intrinsic Value')
        ]
        self.all_stocks_table = DataTable(source=self.data.ticker_list_source, columns=columns, height=800, autosize_mode='fit_viewport', index_position=None, margin=(5, 5, 5, 25))

    def plot_stock_metrics_table(self):
        columns = [
            TableColumn(field='stock', title='Stock'),
            TableColumn(field='intrinsic_value', title='Intrinsic Value'),
            TableColumn(field='intrinsic_value_per_share', title='Intrinsic Value Per Share')
        ]
        self.stock_metrics_table = DataTable(source=self.data.stock_metrics_source, columns=columns, height=800, autosize_mode='fit_viewport', index_position=None, margin=(5, 5, 5, 25), editable=True)

    def plot_income_statement_table(self):
        #  available columns
        #  __________________________________________________________________________________________
        #  key, fiscal_year, time_frame, revenue, COGS, gross_income, SGA, EBIT,
        #  gross_interest_expense, pretax_income, income_tax, net_income, shareholder_net_income,
        #  consolidated_net_income, operating_income, EPS_basic, EPS_diluted

        columns = [
            TableColumn(field='fiscal_year', title='fiscal year'),
            TableColumn(field='revenue', title='revenue'),
            TableColumn(field='COGS', title='COGS'),
            TableColumn(field='gross_income', title='gross income'),
            TableColumn(field='SGA', title='SGA'),
            TableColumn(field='EBIT', title='EBIT'),
            TableColumn(field='gross_interest_expense', title='gross interest expense'),
            TableColumn(field='pretax_income', title='pretax income'),
            TableColumn(field='income_tax', title='income tax'),
            TableColumn(field='net_income', title='net income'),
            TableColumn(field='shareholder_net_income', title='shareholder net income'),
            TableColumn(field='consolidated_net_income', title='consolidated net income'),
            TableColumn(field='operating_income', title='operating income'),
            TableColumn(field='EPS_basic', title='EPS basic'),
            TableColumn(field='EPS_diluted', title='EPS diluted')
        ]
        self.income_statement_table = DataTable(source=self.data.fundamental_source, columns=columns, height=800, autosize_mode='fit_viewport', css_classes=["table_rows"], index_position=None, margin=(5, 5, 5, 25))

    def plot_balance_sheet_table(self):
        #  available columns
        #  __________________________________________________________________________________________
        #  key, fiscal_year, time_frame, total_current_assets,
        #  total_noncurrent_assets, fixed_assets, total_assets, total_current_liabilities,
        #  total_noncurrent_liabilities, total_liabilities, common_equity, total_shareholders_equity,
        #  liabilities_and_shareholder_equity

        columns = [
            TableColumn(field='fiscal_year', title='fiscal year'),
            TableColumn(field='total_current_assets', title='total current assets'),
            TableColumn(field='fixed_assets', title='fixed assets'),
            TableColumn(field='total_assets', title='total assets'),
            TableColumn(field='total_current_liabilities', title='total current liabilities'),
            TableColumn(field='total_noncurrent_liabilities', title='total noncurrent liabilities'),
            TableColumn(field='total_liabilities', title='total liabilities'),
            TableColumn(field='common_equity', title='common equity'),
            TableColumn(field='total_shareholders_equity', title='total shareholders equity'),
            TableColumn(field='liabilities_and_shareholder_equity', title='liabilities and shareholder equity')
        ]
        self.balance_sheet_table = DataTable(source=self.data.fundamental_source, columns=columns, height=800, autosize_mode='fit_viewport', css_classes=["table_rows"], index_position=None, margin=(5, 5, 5, 25))

    def plot_cash_flow_table(self):
        #  available columns
        #  __________________________________________________________________________________________
        #  key, fiscal_year, time_frame, operating_net_cash_flow, investing_net_cash_flow,
        #  financing_net_cash_flow, total_net_cash_flow

        columns = [
            TableColumn(field='fiscal_year', title='fiscal year'),
            TableColumn(field='operating_net_cash_flow', title='operating net cash flow'),
            TableColumn(field='investing_net_cash_flow', title='investing net cash flow'),
            TableColumn(field='financing_net_cash_flow', title='financing net cash flow'),
            TableColumn(field='total_net_cash_flow', title='total net cash flow')
        ]
        self.cash_flow_table = DataTable(source=self.data.fundamental_source, columns=columns, height=800, autosize_mode='fit_viewport', css_classes=["table_rows"], index_position=None, margin=(5, 5, 5, 25))

    def plot_stock_screener_table(self):
        pass

    def plot_income_statement_charts(self):
        pass

    #     # add ADX indicator
    #     ADX = self.ADX_fig.line('seq', 'ADX', source=self.stock_source, line_color='#faf0e6')
    #     pos_DI = self.ADX_fig.line('seq', '+DI', source=self.stock_source, line_color='#32cd32')
    #     neg_DI = self.ADX_fig.line('seq', '-DI', source=self.stock_source, line_color='#ff0800')
    #     # hline_20 = Span(location=20, dimension='height', line_color='silver', line_dash='dotdash', line_width=2)
    #     hline_20 = Span(location=20, dimension='width', line_color='#6A5ACD', line_dash='dashed', line_width=2)
    #
    #     self.ADX_fig.add_layout(hline_20)
    #     self.plot_buysell_lines(self.ADX_fig, 'ADX')
    #
    #     self.tech_indicator_plots.append(self.ADX_fig)
    #     self.glyphs.extend([ADX, pos_DI, neg_DI])
    #
    #     # text box to show which date is currently being hovered over on chart
    #     # self.date_hover_label = Label(x=83, y=0, x_units='data', y_units='data',
    #     #                               text='Date: 2021-02-07', text_color="white", text_align='center', text_font_size='16px', render_mode='css',
    #     #                               background_fill_color='#d3d3d3', background_fill_alpha=0.4)
    #     # self.ADX_fig.add_layout(self.date_hover_label)
    #
    #     self.ADX_fig.xaxis.visible = False
    #
    #     p.vbar(x=[1, 2, 3], width=0.5, bottom=0,
    #            top=[1.2, 2.5, 3.7], color="firebrick")

    def add_interactive_tools(self):
        # date slicer to select start/end date
        start_date_slicer = DatePicker(title='Start Time', value=self.data.start_date, min_date=self.stock.valid_start_date, max_date=self.data.end_date, width=115, height=50, margin=(5, 5, 5, 788))
        start_date_slicer.on_change("value", self.start_date_change)
        self.start_date_slicer = start_date_slicer

        end_date_slicer = DatePicker(title='End Time', value=self.data.end_date, min_date=self.data.start_date, max_date=self.stock.valid_end_date, width=115, height=50, margin=(5, 5, 5, 20))
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

        self.GR_spinner = Spinner(low=0, high=25, value=5, step=.1, width=60, margin=(5, 5, 5, 60), title="Company's Growth Rate (%)")
        self.GR_spinner.on_change("value", self.update_GR)

        # add button to export stock data to excel
        self.export_excel_button = Button(label="Export to Excel", button_type="primary", width=55, margin=(17, 15, 5, 20))
        self.export_excel_button.on_click(self.export_to_excel)
        # self.export_to_excel()

    def draw_line_event(self):
        return CustomJS(args=dict(source=self.data.draw_lines_source, button_source=self.data.button_source), code="""
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
        return CustomJS(args=dict(source=self.data.draw_rect_source, button_source=self.data.button_source, x_range=self.candlestick_fig.x_range, y_limits=(self.candlestick_fig.y_range.start, self.candlestick_fig.y_range.end)), code="""
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
        in_range = x >= 0 and x < len(self.data.df)

        if cb_obj.event_name == 'mouseleave' or not in_range:
            self.date_hover_label.text = ""
            self.candlestick_fig.title.text = ""

        if not in_range:
            return

        date = self.data.df.loc[x, 'date'].date()
        open_price = self.data.df.loc[x, 'open']
        high_price = self.data.df.loc[x, 'high']
        low_price = self.data.df.loc[x, 'low']
        close_price = self.data.df.loc[x, 'close']
        percent_change = self.data.df.loc[x, 'percent_change']

        if cb_obj.event_name == 'mouseenter' or cb_obj.event_name == 'mousemove':
            # change date label to move with hovering cursor
            self.date_hover_label.text = f'{date}'
            self.date_hover_label.x = x

            title = f"{date}\tO: {open_price}   H: {high_price}   L: {low_price}   C: {close_price} ({percent_change}%)"
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
        if not (new_date >= self.data.start_date and new_date <= self.data.end_date):
            stock = Stock(self.stock.ticker, self.stock.SQL, self.trading_calendar, self.logger, request_start_date=new_date, request_end_date=self.data.end_date)
            self.init_stock(stock=stock)
            self.reset_charts()
            self.replot()

        date_index = self.date_to_index[new_date]
        x_range = self.candlestick_fig.x_range
        x_range.start = date_index

    def end_date_change(self, attr, old_date, new_date):
        # if new end date is not within valid range of dates, then recreate stock obj with new start/end params.
        new_date = convert_str_to_date(new_date)
        if not (new_date >= self.data.start_date and new_date <= self.data.end_date):
            stock = Stock(self.stock.ticker, self.stock.SQL, self.trading_calendar, self.logger, request_start_date=self.data.start_date, request_end_date=new_date)
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

    def init_events(self):
        point_events = [Pan, PanStart, PanEnd]

        events = [Pan, PanStart, PanEnd]
        for event in events:
            self.candlestick_fig.js_on_event(event, self.draw_rect_event())
            self.candlestick_fig.js_on_event(event, self.draw_line_event())

        events = [MouseMove, MouseEnter, MouseLeave]
        for event in events:
            self.candlestick_fig.on_event(event, self.show_hover_prices)

        events = MouseEnter

        self.candlestick_fig.on_event(MouseWheel, self.move_chart_event)

    def button_callback(self):
        pass

    def init_button(self):
        self.button = Button(label="GFG")
        self.button.js_on_click(
            CustomJS(args=dict(source=self.data.button_source), code="""
                const data = source.data

                data['value'][0] = !data['value'][0]

                console.log(data['value'][0])
                source.change.emit()            
            """)
        )

    def init_autoscale(self):
        self.candlestick_fig.x_range.on_change('start', self.autoscale_candlestick_yaxis)
        self.candlestick_fig.x_range.on_change('end', self.autoscale_candlestick_yaxis)

    def autoscale_candlestick_yaxis(self, attr, old, new):
        # source = ColumnDataSource({'date': self.data.df.date, 'high': self.data.df.high, 'low': self.data.df.low, 'index': [i for i in range(len(self.data.df.date))]})
        # pdb.set_trace()
        # breakpoint()
        index = [i for i in range(len(self.data.df.date))]
        high = self.data.df.high
        low = self.data.df.low
        volume = self.data.df.volume
        # active_tech_indicators = [self.tech_indicator_plots[i].name for i in self.checkbox_button_group.active]
        # tech_ind_data = [self.data.df[ind] for ind in active_tech_indicators if ind is not None]
        active_tech_indicators = [self.tech_indicator_plots[i].name for i in range(len(self.tech_indicator_plots)) if self.tech_indicator_plots[i].visible]
        tech_ind_data = [self.data.df[ind] for ind in active_tech_indicators if ind is not None]

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
        top_padding = (max_val - min_val) * 0.25

        y_range.start = min_val - bottom_padding - (pad / 2)
        y_range.end = max_val + top_padding + pad

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
        self.autoscale_ADX_yaxis(1, 1, 1)
        self.autoscale_buysell_ind_yaxis(1, 1, 1)
        self.autoscale_buysell_strat_yaxis(1, 1, 1)

    def remove_prev_hover_tool(self):
        for tool in self.candlestick_fig.toolbar.tools:
            if isinstance(tool, tools.HoverTool):
                self.candlestick_fig.toolbar.tools.remove(tool)

    def get_html_formatter(self):
        template = """
        <div style="color:<%=percent_change_color%>";>
        <%= value %>
        </div>
        """
        return HTMLTemplateFormatter(template=template)

    # plots vertical lines at buy/sell locations (does not print
    def plot_buysell_lines(self, fig, ind):
        if self.data.buysell_results_dict[ind] is None:
            return

        dates = self.data.buysell_results_dict[ind].data['dates']
        buy_list = self.data.buysell_results_dict[ind].data['buy']
        sell_list = self.data.buysell_results_dict[ind].data['sell']

        # if ind == 'EMA2':
        #     breakpoint()

        vlines_plot = []
        vlines_queue = []
        for date, buy, sell in zip(dates, buy_list, sell_list):
            if isinstance(date, datetime.date):
                # if str(date) == '2019-12-03':
                #     breakpoint()

                date_index = self.date_to_index[date]

                if is_number(buy):
                    vline = Span(location=date_index, dimension='height', line_color='#93c47d', line_dash='dotdash', line_width=2)
                else:
                    assert (is_number(sell))
                    vline = Span(location=date_index, dimension='height', line_color='#DB575E', line_dash='dotdash', line_width=2)

                vline.visible = False
                vlines_queue.append((date_index, vline))
                if len(vlines_queue) == 2:
                    i2, vline2 = vlines_queue.pop()
                    i1, vline1 = vlines_queue.pop()
                    if (i2 - i1) != 1:
                        fig.add_layout(vline1)
                        fig.add_layout(vline2)
                        if fig == self.candlestick_fig:
                            vline1.visible = False
                            vline2.visible = False
                            vlines_plot.extend([vline1, vline2])

        if fig == self.candlestick_fig:
            self.main_chart_buysell_lines.append(vlines_plot)

    def plot_profit_loss_ranges(self, fig, ind):

        if ind not in self.data.buysell_results_dict or self.data.buysell_results_dict[ind] is None:
            return

        buy_indexes = np.argwhere(~np.isnan(self.data.buysell_results_dict[ind]['buy']).to_numpy())
        sell_indexes = np.argwhere(~np.isnan(self.data.buysell_results_dict[ind]['sell']).to_numpy())
        dates = self.data.buysell_results_dict[ind]['dates']
        buy_prices = self.data.buysell_results_dict[ind]['buy']
        sell_prices = self.data.buysell_results_dict[ind]['sell']

        glyph_boxes = []
        for buy_i, sell_i in zip(np.nditer(buy_indexes), np.nditer(sell_indexes)):
            buy_i, sell_i = int(buy_i), int(sell_i)

            buy_date = dates[buy_i]
            sell_date = dates[sell_i]
            buy_price = buy_prices[buy_i]
            sell_price = sell_prices[sell_i]

            profit = sell_price - buy_price

            if profit >= 0:
                profit_box = BoxAnnotation(left=buy_i, right=sell_i, fill_alpha=0.2, fill_color=self.buy_color)
                # profit_box = BoxAnnotation(left=buy_i, right=sell_i, fill_alpha=0.2, fill_color='#2ECC71')

            else:
                profit_box = BoxAnnotation(left=buy_i, right=sell_i, fill_alpha=0.2, fill_color=self.sell_color)
                # profit_box = BoxAnnotation(left=buy_i, right=sell_i, fill_alpha=0.2, fill_color='#ED2939')

            profit_box.visible = False
            fig.add_layout(profit_box)
            glyph_boxes.append(profit_box)

        return glyph_boxes

    # def plot_all_buysell_scatter_charts(self, fig):
    #     # -- add strat --
    #     self.plot_buysell_scatter_chart(fig, 'minmax')
    #     self.plot_buysell_scatter_chart(fig, 'MACD')
    #     self.plot_buysell_scatter_chart(fig, 'RSI')
    #     self.plot_buysell_scatter_chart(fig, 'EMA2')
    #     self.plot_buysell_scatter_chart(fig, 'EMA3')
    #     self.plot_buysell_scatter_chart(fig, 'ADX')
    #     self.plot_buysell_scatter_chart(fig, 'ADX2')

    def plot_buysell_scatter_chart(self, fig, ind):

        buy_chart = fig.scatter('seq', 'buy', marker="circle", source=self.data.buysell_results_dict[ind], color="green")
        sell_chart = fig.scatter('seq', 'sell', marker="circle", source=self.data.buysell_results_dict[ind], color="red")
        buy_chart.visible = False
        sell_chart.visible = False

        return (buy_chart, sell_chart)

    #
    # def change_tab(self, attr, old, new):
    #     active_tab = new
    #     valid_tab_len = len(self.profit_loss_ranges)-1
    #     # print(f'prev tab: {self.prev_tab}, active tab: {active_tab}')
    #
    #     # if user clicks on a non tech indicator chart, like fundamental data, then do nothing
    #     if active_tab and active_tab > valid_tab_len:
    #         return
    #
    #     # can occur if user clicks tab outside of valid tab len then back to prev tab
    #     if active_tab == self.prev_tab:
    #         return
    #
    #     if self.prev_tab is not None:
    #         # for buysell_line in self.main_chart_buysell_lines[prev_tab]:
    #         #     buysell_line.visible = False
    #
    #         for profit_loss_box in self.profit_loss_ranges[self.prev_tab]:
    #             profit_loss_box.visible = False
    #
    #         for profit_loss_box in self.buysell_scatter_charts[self.prev_tab]:
    #             profit_loss_box.visible = False
    #
    #     # for buysell_line in self.main_chart_buysell_lines[active_tab]:
    #     #     buysell_line.visible = True
    #
    #     for profit_loss_box in self.profit_loss_ranges[active_tab]:
    #         profit_loss_box.visible = True
    #
    #     for scatter_chart in self.buysell_scatter_charts[active_tab]:
    #         scatter_chart.visible = True
    #
    #     self.prev_tab = active_tab

    def init_google_sheets_API(self):
        # breakpoint()
        SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('/Users/landon/PycharmProjects/Stocks/stock_package/backtrader/stock-data-project-339820-f892045254d5.json', SCOPES)
        file = gspread.authorize(creds)
        self.excel_doc = file.open("Stock_Data_Analysis_v2")
        # sheet = sheet.sheet_name  #replace sheet_name with the name that corresponds to yours, e.g, it can be sheet1
        self.excel_sheet = self.excel_doc.worksheet('raw_data')

    def export_to_excel(self):
        start_time = time.time()

        df = self.data.df.copy()
        # breakpoint()
        df['date'] = df['date'].dt.strftime('%m/%d/%Y')
        self.excel_sheet.update('A1', [list(df.keys())])
        self.excel_sheet.update('A2', df.values.tolist())

        # export buysell indicators too
        df2 = self.buysell_df_dict
        self.excel_sheet = self.excel_doc.worksheet('buysell_data')

        end_time = time.time()
        print(f'excel took: {end_time - start_time}')

        col_names = []
        merge_dfs = None
        df2_copy = df2.copy()

        for ind, tmp_df in df2.items():
            # breakpoint()
            df2_copy[ind] = df2_copy[ind].fillna(0)
            df2_copy[ind]['dates'] = [d.strftime('%m/%d/%Y') if isinstance(d, datetime.date) else d for d in df2_copy[ind]['dates'].values.tolist()]

            # delete unncessary columns
            if 'data' in df2_copy[ind]:
                df2_copy[ind] = df2_copy[ind].drop(columns='data')
            if 'seq' in df2_copy[ind]:
                df2_copy[ind] = df2_copy[ind].drop(columns='seq')

            # change column names so merged df doesn't have duplicate col names
            new_names = {'buy': f'{ind} buy', 'sell': f'{ind} sell', 'dates': f'{ind} dates'}
            df2_copy[ind] = df2_copy[ind].rename(index=str, columns=new_names)

            # breakpoint()
            col_names.extend(df2_copy[ind].columns)
            if merge_dfs is None:
                merge_dfs = df2_copy[ind]
            else:
                merge_dfs = merge_dfs.join(df2_copy[ind])

        # breakpoint()
        self.excel_sheet.update('A1', [col_names])
        self.excel_sheet.update('A2', merge_dfs.values.tolist())


if __name__ == '__main__':
    pass
