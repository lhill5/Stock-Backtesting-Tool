import pandas as pd
from math import pi, sqrt
from bokeh.plotting import figure, show, output_file, curdoc, reset_output
from bokeh.models import tools, HoverTool, ColumnDataSource, CustomJS, Button, Rect, Span, CheckboxButtonGroup, DatePicker, MultiChoice, Range1d, LinearAxis
from bokeh.events import ButtonClick, MouseMove, PanStart, PanEnd, Pan, MouseWheel, MouseEnter
from bokeh.layouts import row, column, gridplot
from bokeh.palettes import RdYlBu3
from bokeh.themes import built_in_themes
import datetime
import numpy as np
import pdb
from Stock import Stock
import math
from financial_calcs import convert_str_to_date, divide



class Graph:
    def __init__(self, stock, stocks):
        self.stocks = stocks
        self.candlestick_fig = None
        # self.volume_fig = None
        self.MACD_fig = None
        self.RSI_fig = None

        self.init_stock(stock)


    def init_stock(self, stock=None, ticker=None):
        # must pass either a "Stock" object or ticker name in order to initialize the currently graphed stock
        if stock is None and ticker is None:
            return

        if ticker:
            start_date, end_date = datetime.date(2019, 1, 1), datetime.date(2020, 1, 1)
            stock = Stock(ticker, self.stock.SQL, start_date=start_date, end_date=end_date)

        self.stock = stock
        self.stock_name = stock.ticker
        self.stock_dict = stock.stock_dict
        self.tech_indicators = stock.tech_indicators
        self.date_to_index = stock.date_to_index

        self.moving_averages = self.tech_indicators['EMA'].keys()
        self.df = self.transform_data()

        self.len = self.df.shape[0]
        self.start, self.end = 0, self.len - 1
        self.start_date, self.end_date = convert_str_to_date(self.stock_dict['date'][0]), convert_str_to_date(self.stock_dict['date'][-1])
        self.y_limits = self.get_starting_y_limits()

        # user for interactivity tools (draw support/resistance lines, draw drag-highlight feature to zoom into plot area, click button to toggle between draw lines / zoom features)
        self.draw_lines_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.draw_rect_source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[]))
        self.button_source = ColumnDataSource(data=dict(value=[True]))

        self.wheel_prev_x = -1

        # creates bokeh figures if they don't already exist
        if self.candlestick_fig is None and self.MACD_fig is None and self.RSI_fig is None:
            TOOLS = 'wheel_zoom, reset, save'
            self.candlestick_fig = figure(x_axis_type='linear', width=1400, height=400, toolbar_location="right", tools=TOOLS, x_range=(self.start, self.end), y_range=(self.y_limits[0], self.y_limits[1]))
            # self.volume_fig = figure(x_axis_type='linear', width=1400, height=100, toolbar_location="right", tools=TOOLS, x_range=(self.start, self.end))
            self.MACD_fig = figure(x_axis_type='linear', tools="", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range)
            self.RSI_fig = figure(x_axis_type='linear', tools="", toolbar_location=None, width=1400, height=200, x_range=self.candlestick_fig.x_range, y_range=(0, 100))

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


    def plot(self):
        df = self.df
        w=0.3

        seqs=np.arange(self.len)
        df["seq"]=pd.Series(seqs)
        df["date"] = pd.to_datetime(df["date"])

        inc = df.close > df.open
        dec = df.open > df.close
        source = ColumnDataSource(ColumnDataSource.from_df(df))
        self.sourceInc=ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
        self.sourceDec=ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))

        hist_inc = df.histogram > 0
        hist_dec = df.histogram <= 0
        histogramInc = ColumnDataSource(ColumnDataSource.from_df(df.loc[hist_inc]))
        histogramDec = ColumnDataSource(ColumnDataSource.from_df(df.loc[hist_dec]))

        #the values for the tooltip come from ColumnDataSource
        hover = HoverTool(
            names=[
                "green_candle",
                "red_candle"
            ],
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

        # if replotting this graph, this will remove previous stock/start-end range hover tool, and the updated new hover tool
        self.remove_prev_hover_tool()
        self.candlestick_fig.add_tools(hover)

        # p = self.candlestick_fig
        self.candlestick_fig.xaxis.major_label_orientation = pi/4
        self.candlestick_fig.grid.grid_line_alpha=0.3

        # records all glyphs to be modified/changed later (used in change_stock function when plotting new stock data)
        self.glyphs = []

        # creates highlight box drawn by user
        rect = Rect(x='x', y='y', width='width', height='height',
                    fill_alpha=0.3, fill_color='#009933')
        rect_glyph = self.candlestick_fig.add_glyph(self.draw_rect_source, rect, selection_glyph=rect, nonselection_glyph=rect)
        self.glyphs.append(rect_glyph)

        # creates lines drawn by user (can be used to draw support/resistance levels)
        line_glyph = self.candlestick_fig.multi_line('x', 'y', line_width=2, source=self.draw_lines_source)
        self.glyphs.append(line_glyph)

        # candlestick drawing
        candle_upline = self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color="#12C98C", source=self.sourceInc)
        candle_downline = self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color="#F2583E", source=self.sourceDec)
        candle_up = self.candlestick_fig.vbar('seq', w, 'open', 'close', fill_color="#12C98C", line_color="#12C98C", source=self.sourceInc, name="green_candle")
        candle_down = self.candlestick_fig.vbar('seq', w, 'open', 'close', fill_color="#F2583E", line_color="#F2583E", source=self.sourceDec, name="red_candle")
        self.glyphs.extend([candle_upline, candle_downline, candle_up, candle_down])

        # add volume to candlestick chart
        #____________________________________________________________________________________________________#
        self.candlestick_fig.extra_y_ranges = {"volume_axis": Range1d(start=0, end=max(self.df['volume'])*5)}
        # removes previous second y-axes (happens after start/end date is changed or stock is changed and replot(...) is called)
        self.candlestick_fig.right = []

        self.candlestick_fig.add_layout(LinearAxis(y_range_name="volume_axis"), 'right')
        volume_up = self.candlestick_fig.vbar('seq', w-0.1, 0, 'volume', source=self.sourceInc, y_range_name='volume_axis', fill_color = "rgba(18, 201, 140, 0.35)", line_color = "rgba(18, 201, 140, 0.35)")
        volume_down = self.candlestick_fig.vbar('seq', w-0.1, 0, 'volume', source=self.sourceDec, y_range_name='volume_axis', fill_color = "rgba(242, 88, 62, 0.35)", line_color = "rgba(242, 88, 62, 0.35)")
        self.glyphs.extend([volume_up, volume_down])

        # add technical indicators
        self.tech_indicator_plots = []

        # add moving averages to plot
        moving_avg_colors = ['orange', 'green', 'purple', 'yellow']
        EMAs = [self.candlestick_fig.line('seq', f'EMA_{avg_num}', source=source, line_color=moving_avg_colors[i], name=f"EMA_{avg_num}") for i, avg_num in enumerate(self.moving_averages)]
        SMAs = [self.candlestick_fig.line('seq', f'SMA_{avg_num}', source=source, line_color=moving_avg_colors[i], name=f"SMA_{avg_num}") for i, avg_num in enumerate(self.moving_averages)]
        self.tech_indicator_plots.extend(EMAs)
        self.tech_indicator_plots.extend(SMAs)
        self.glyphs.extend(EMAs)
        self.glyphs.extend(SMAs)

        # add MACD indicators
        histogram_up = self.MACD_fig.vbar('seq', w, 'histogram', source=histogramInc, fill_color="#12C98C", line_color="#12C98C")
        histogram_down = self.MACD_fig.vbar('seq', w, 'histogram', source=histogramDec, fill_color = "#F2583E", line_color = "#F2583E")
        macd = self.MACD_fig.line('seq', 'MACD', source=source, line_color='orange')
        signal = self.MACD_fig.line('seq', 'signal', source=source, line_color='#5985FF')
        self.tech_indicator_plots.append(self.MACD_fig)
        self.glyphs.extend([histogram_up, histogram_down, macd, signal])

        # add RSI indicator
        rsi = self.RSI_fig.line('seq', 'RSI', source=source, line_color='#04BFDC')
        hline_30 = Span(location=30, dimension='width', line_color='red', line_dash='solid', line_width=2)
        hline_70 = Span(location=70, dimension='width', line_color='red', line_dash='solid', line_width=2)
        self.RSI_fig.add_layout(hline_30)
        self.RSI_fig.add_layout(hline_70)
        self.tech_indicator_plots.append(self.RSI_fig)
        self.glyphs.append(rsi)

        self.button_labels = ['EMA 9', 'EMA 50', 'EMA 150', 'EMA 200', 'SMA 9', 'SMA 50', 'SMA 150', 'SMA 200', 'MACD', 'RSI']
        default_active_plots = [0, 1, 3, 8, 9] # add 9 for RSI
        for i, tech_ind in enumerate(self.tech_indicator_plots):
            tech_ind.visible = i in default_active_plots

        # date slicer to select start/end date
        start_date_slicer = DatePicker(title='Start Time', value=self.start_date, min_date=self.stock.valid_start_date_offset, max_date=self.end_date, width=115, height=50, margin=(5,5,5,5))
        start_date_slicer.on_change("value", self.start_date_change)

        end_date_slicer = DatePicker(title='End Time', value=self.end_date, min_date=self.start_date, max_date=self.stock.valid_end_date, width=115, height=50, margin=(5,5,5,10))
        end_date_slicer.on_change("value", self.end_date_change)

        # buttons to select technical indicators to show/hide
        self.checkbox_button_group = CheckboxButtonGroup(labels=self.button_labels, active=default_active_plots, button_type="primary", css_classes=["button_margin"], height=30, margin=(23,5,5,20))
        self.checkbox_button_group.on_click(self.click_tech_indicators_button)

        OPTIONS = self.stocks
        self.select_stock = MultiChoice(value=['MSFT'], options=OPTIONS, css_classes=["stock_picker_color"], width=210, max_items=1, margin=(17,15,5,35))
        self.select_stock.on_change("value", self.stock_change)

        # remove x-axis labels
        # pdb.set_trace()
        self.candlestick_fig.xaxis.visible = False
        self.candlestick_fig.yaxis[-1].visible = False
        self.MACD_fig.xaxis.visible = False

        # autoscale axes when first plotting stock (accounts for padding between candlestick chart and volume bars)
        self.autoscale_candlestick_yaxis(1, 1, 1)
        self.autoscale_MACD_yaxis(1, 1, 1)

        # layout = gridplot([[row(self.select_stock, start_date_slicer, end_date_slicer, self.checkbox_button_group)], [self.candlestick_fig]])
        layout = column(row(self.select_stock, start_date_slicer, end_date_slicer, self.checkbox_button_group), self.candlestick_fig, self.MACD_fig, self.RSI_fig)
        curdoc().add_root(layout)


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


    def move_chart_event(self, cb_obj):

        x_range = self.candlestick_fig.x_range
        delta = 1 if cb_obj.delta > 0 else -1

        if self.wheel_prev_x == -1:
            self.wheel_prev_x = cb_obj.sx
            print('wheel x is -1')
            return

        move_speed = round((0.0175 * (x_range.end - x_range.start)), 1)
        print(f'x start: {round(x_range.start,1)}, x end: {round(x_range.end,1)}, move speed: {move_speed}')
        # move_speed = 3
        if x_range.start - (move_speed * delta) >= self.start and x_range.end - (move_speed * delta) <= self.end:
            x_range.start -= move_speed * delta
            x_range.end -= move_speed * delta


    def click_tech_indicators_button(self, active_plots):
        # pdb.set_trace()
        # print([self.tech_indicator_plots[i].name for i in self.checkbox_button_group.active])
        for i, tech_ind in enumerate(self.tech_indicator_plots):
            tech_ind.visible = i in active_plots


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


    def init_events(self):
        point_events = [Pan, PanStart, PanEnd]

        events = [Pan, PanStart, PanEnd]
        for event in events:
            # self.candlestick_fig.js_on_event(event, self.draw_rect_event())
            self.candlestick_fig.js_on_event(event, self.draw_rect_event())
            self.candlestick_fig.js_on_event(event, self.draw_line_event())

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


    def autoscale_candlestick_yaxis(self, attr, old, new):
        # source = ColumnDataSource({'date': self.df.date, 'high': self.df.high, 'low': self.df.low, 'index': [i for i in range(len(self.df.date))]})
        # pdb.set_trace()
        index = [i for i in range(len(self.df.date))]
        high = self.df.high
        low = self.df.low
        volume = self.df.volume
        active_tech_indicators = [self.tech_indicator_plots[i].name for i in self.checkbox_button_group.active]
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
    def transform_data(self):
        stock_data = {key: vals for key, vals in self.stock_dict.items()}
        df = pd.DataFrame.from_dict(stock_data)

        # add columns to df
        # df['EMA_9'] = self.tech_indicators['EMA'][9]
        # df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'EMA_9']
        df.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']

        # transform date to pandas timestamp (must be this otherwise data doesn't showup on bokeh chart)
        df['date'] = pd.to_datetime([x for x in df['date'].squeeze().tolist()], dayfirst=True)

        # add new columns (technical indicators)
        for indicator, values in self.tech_indicators.items():
            if 'EMA' in indicator or 'SMA' in indicator:
                for num_days, val in values.items():
                    df[f'{indicator}_{num_days}'] = val
            else:
                df[indicator] = values

        return df


    def restart_storage_vars(self):
        return CustomJS(code="""
            localStorage.setItem('prev_x_val', -1);
        """)


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


    def remove_prev_hover_tool(self):
        for tool in self.candlestick_fig.toolbar.tools:
            if isinstance(tool, tools.HoverTool):
                self.candlestick_fig.toolbar.tools.remove(tool)


if __name__ == '__main__':
    pass

