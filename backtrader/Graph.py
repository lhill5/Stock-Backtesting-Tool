import pandas as pd
from math import pi, sqrt
from bokeh.plotting import figure, show, output_file, curdoc
from bokeh.models import HoverTool, ColumnDataSource, CustomJS, Button, Rect, Span, CheckboxButtonGroup
from bokeh.events import ButtonClick, MouseMove, PanStart, PanEnd, Pan, MouseWheel, MouseEnter
from bokeh.layouts import column, gridplot
from bokeh.palettes import RdYlBu3
from bokeh.themes import built_in_themes
import datetime
import numpy as np
import pdb


class Graph:
    def __init__(self, stock, stock_dict, tech_indicators, date_to_index):
        self.stock_name = stock
        self.stock_dict = stock_dict
        self.tech_indicators = tech_indicators
        self.date_to_index = date_to_index
        self.moving_averages = self.tech_indicators['EMA'].keys()
        self.df = self.transform_data()

        self.len = self.df.shape[0]
        self.start, self.end = 0, self.len - 1
        # self.candlestick_fig = figure(x_axis_type='datetime', plot_width=1200, title='test', tools="xpan, wheel_zoom, reset, help")
        # self.end = datetime.date(2021, 7, 6)
        # self.start = (self.end - datetime.timedelta(days=12*31))
        self.y_limits = self.get_starting_y_limits()
        # print(self.y_limits)
        print(self.start, self.end)
        # curdoc().theme = 'dark_minimal'
        TOOLS = 'wheel_zoom, reset, save'
        self.candlestick_fig = figure(x_axis_type='linear', plot_width=1400, plot_height=400, title='test', tools=TOOLS, x_range=(self.start, self.end), y_range=(self.y_limits[0], self.y_limits[1]))
        self.MACD_fig = figure(x_axis_type='linear', plot_width=1400, plot_height=200, tools=TOOLS, x_range=self.candlestick_fig.x_range)
        self.RSI_fig = figure(x_axis_type='linear', plot_width=1400, plot_height=200, tools=TOOLS, x_range=self.candlestick_fig.x_range, y_range=(0, 100))

        # self.candlestick_fig = figure(plot_width=1000, plot_height = 600, title='test', tools=TOOLS, y_range=(self.y_limits[0], self.y_limits[1]))
        # self.candlestick_fig = figure(plot_width=1000, plot_height=600, tools=TOOLS)

        self.draw_lines_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.draw_rect_source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[]))
        self.button_source = ColumnDataSource(data=dict(value=[True]))

        self.wheel_prev_x = -1

        self.init_events()
        self.init_button()
        self.init_autoscale()


    def plot(self):
        df = self.df

        seqs=np.arange(self.len)
        df["seq"]=pd.Series(seqs)
        df["date"] = pd.to_datetime(df["date"])

        inc = df.close > df.open
        dec = df.open > df.close
        w=0.3

        #use ColumnDataSource to pass in data for tooltips
        source = ColumnDataSource(ColumnDataSource.from_df(df))
        sourceInc=ColumnDataSource(ColumnDataSource.from_df(df.loc[inc]))
        sourceDec=ColumnDataSource(ColumnDataSource.from_df(df.loc[dec]))

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

        # TOOLS = 'wheel_zoom,box_zoom,reset,save,crosshair'
        # p = figure(plot_width=1200, plot_height=600, tools=TOOLS)
        self.candlestick_fig.add_tools(hover)

        # p = self.candlestick_fig
        self.candlestick_fig.xaxis.major_label_orientation = pi/4
        self.candlestick_fig.grid.grid_line_alpha=0.3

        # creates highlight box drawn by user
        rect = Rect(x='x', y='y', width='width', height='height',
                    fill_alpha=0.3, fill_color='#009933')
        self.candlestick_fig.add_glyph(self.draw_rect_source, rect, selection_glyph=rect, nonselection_glyph=rect)

        # creates lines drawn by user
        self.candlestick_fig.multi_line('x', 'y', line_width=2, source=self.draw_lines_source)

        # candlestick drawing
        self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color="#12C98C", source=sourceInc)
        self.candlestick_fig.segment('seq', 'high', 'seq', 'low', color="#F2583E", source=sourceDec)
        self.candlestick_fig.vbar('seq', w, 'open', 'close', fill_color="#12C98C", line_color="#12C98C", source=sourceInc, name="green_candle")
        self.candlestick_fig.vbar('seq', w, 'open', 'close', fill_color="#F2583E", line_color="#F2583E", source=sourceDec, name="red_candle")

        # add technical indicators
        self.tech_indicator_plots = []

        # add moving averages to plot
        moving_avg_colors = ['orange', 'green', 'purple', 'yellow']
        EMAs = [self.candlestick_fig.line('seq', f'EMA_{avg_num}', source=source, line_color=moving_avg_colors[i]) for i, avg_num in enumerate(self.moving_averages)]
        SMAs = [self.candlestick_fig.line('seq', f'SMA_{avg_num}', source=source, line_color=moving_avg_colors[i]) for i, avg_num in enumerate(self.moving_averages)]
        self.tech_indicator_plots.extend(EMAs)
        self.tech_indicator_plots.extend(SMAs)

        # add MACD indicators
        self.MACD_fig.line('seq', 'MACD', source=source, line_color='orange')
        self.MACD_fig.line('seq', 'signal', source=source, line_color='#5985FF')
        self.tech_indicator_plots.append(self.MACD_fig)

        # add RSI indicator
        self.RSI_fig.line('seq', 'RSI', source=source, line_color='#04BFDC')
        hline_30 = Span(location=30, dimension='width', line_color='red', line_dash='solid', line_width=2)
        hline_70 = Span(location=70, dimension='width', line_color='red', line_dash='solid', line_width=2)
        self.RSI_fig.add_layout(hline_30)
        self.RSI_fig.add_layout(hline_70)
        self.tech_indicator_plots.append(self.RSI_fig)

        self.button_labels = ['EMA 9', 'EMA 50', 'EMA 150', 'EMA 200', 'SMA 9', 'SMA 50', 'SMA 150', 'SMA 200', 'MACD', 'RSI']
        default_active_plots = [0, 1, 3, 8, 9]
        for i, tech_ind in enumerate(self.tech_indicator_plots):
            tech_ind.visible = i in default_active_plots

        checkbox_button_group = CheckboxButtonGroup(labels=self.button_labels, active=default_active_plots, button_type="primary", css_classes=["button_margin"])
        checkbox_button_group.on_click(self.click_tech_indicators_button)

        layout = gridplot([[checkbox_button_group], [self.candlestick_fig], [self.MACD_fig], [self.RSI_fig]])

        # remove x-axis labels
        self.candlestick_fig.xaxis.visible = False
        self.MACD_fig.xaxis.visible = False

        # show(layout)
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

        print(x_range.start, x_range.end)

        delta = 1 if cb_obj.delta > 0 else -1
        if self.wheel_prev_x == -1:
            self.wheel_prev_x = cb_obj.sx
            print('wheel x is -1')
            return

        # elif cb_obj.sx != self.wheel_prev_x:
        #     print(cb_obj.sx, self.wheel_prev_x)
        #     print('wheel x isn\'t equal to sx')
        #     return
        move_speed = round((0.0175 * (x_range.end - x_range.start)), 1)
        print(x_range.end - x_range.start, move_speed)
        if x_range.start - (move_speed * delta) >= self.start and x_range.end - (move_speed * delta) <= self.end:
            x_range.start -= move_speed * delta
            x_range.end -= move_speed * delta

        # return CustomJS(args=dict(x_range=self.candlestick_fig.x_range, x_axis_source=self.x_axis_source), code='''
        #     clearTimeout(window._autoscale_timeout);
        #     const limits = x_axis_source.data;
        #
        #     console.log('mouse scrolled')
        #     console.log('start: ' + x_range.start + ' end: ' + x_range.end)
        #     console.log('start: ' + limits['start'] + ' end: ' + limits['end'])
        #
        #     let delta;
        #     if (cb_obj['delta'] > 0)
        #         delta = 1;
        #     else
        #         delta = -1;
        #
        #     //console.log(delta);
        #
        #     let prev_x = Number(localStorage.getItem('prev_x_val'));
        #     console.log('prev_x: ' + prev_x + ' sx: ' + cb_obj['sx'])
        #     if (prev_x === -1) {
        #         localStorage.setItem('prev_x_val', cb_obj['sx']);
        #         return;
        #     }
        #     // ignore input if user scrolled horizontally (can only scroll vertically)
        #     else if (cb_obj['sx'] !== prev_x) {
        #         //console.log('scrolled horizontally');
        #         return;
        #     }
        #
        #     // updates x_axis start/end values based on which direction user scrolled
        #     limits['start'][0] -= 3 * delta;
        #     limits['end'][0] -= 3 * delta;
        #     x_range.start = limits['start'][0];
        #     x_range.end = limits['end'][0];
        #
        #     source.change.emit();
        # ''')


    def click_tech_indicators_button(self, active_plots):
        # pdb.set_trace()
        for i, tech_ind in enumerate(self.tech_indicator_plots):
            tech_ind.visible = i in active_plots


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
        source = ColumnDataSource({'date': self.df.date, 'high': self.df.high, 'low': self.df.low, 'index': [i for i in range(len(self.df.date))]})

        autoscale_callback = CustomJS(args=dict(x_range=self.candlestick_fig.x_range, y_range=self.candlestick_fig.y_range, y_limits=(self.y_limits[0], self.y_limits[1]), source=source), code='''
            clearTimeout(window._autoscale_timeout);
            
           
            var date = source.data.date,
                index = source.data.index,
                low = source.data.low,
                high = source.data.high,
                start = x_range.start,
                end = x_range.end,
                min = Infinity,
                max = -Infinity;

            for (var i=0; i < index.length; ++i) {  
                if (i >= start && i <= end) {
                    if (high[i] > max) {
                        //console.log(high[i])
                        //console.log(i)
                    }
                    max = Math.max(high[i], max);
                    min = Math.min(low[i], min);
                }
            }
            var pad = (max - min) * 0.05;
            
            window._autoscale_timeout = setTimeout(function() {
                y_range.start = min - pad;
                y_range.end = max + pad;
            });
        ''')

        self.candlestick_fig.x_range.js_on_change('end', autoscale_callback)


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


if __name__ == '__main__':
    pass

