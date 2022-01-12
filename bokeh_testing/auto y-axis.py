import pandas as pd

from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import CustomJS, ColumnDataSource
from stock_package.backtrader.Stock import Stock
import datetime


def candlestick_plot(df):
    fig = figure(sizing_mode='stretch_both',
                 tools='xwheel_zoom',
                 active_scroll='xwheel_zoom',
                 x_axis_type='datetime')

    inc = df.close > df.open
    dec = ~inc

    fig.segment(df.date[inc], df.high[inc], df.date[inc], df.low[inc], color="green")
    fig.segment(df.date[dec], df.high[dec], df.date[dec], df.low[dec], color="red")
    width_ms = 12*60*60*1000 # half day in ms
    fig.vbar(df.date[inc], width_ms, df.open[inc], df.close[inc], color="green")
    fig.vbar(df.date[dec], width_ms, df.open[dec], df.close[dec], color="red")

    source = ColumnDataSource({'date': df.date, 'high': df.high, 'low': df.low})
    autoscale_callback = CustomJS(args=dict(y_range=fig.y_range, source=source), code='''
        clearTimeout(window._autoscale_timeout);

        var date = source.data.date,
            low = source.data.low,
            high = source.data.high,
            start = cb_obj.start,
            end = cb_obj.end,
            min = Infinity,
            max = -Infinity;

        for (var i=0; i < date.length; ++i) {
            console.log(date[i]);
            console.log(start);
            console.log(end);
            if (date[i] >= start && date[i] <= end) {
                max = Math.max(high[i], max);
                min = Math.min(low[i], min);
            }
        }
        var pad = (max - min) * .25;

        window._autoscale_timeout = setTimeout(function() {
            y_range.start = min - pad;
            y_range.end = max + pad;
        });
    ''')

    fig.x_range.js_on_change('start', autoscale_callback)
    show(fig)


if __name__ == '__main__':
    MSFT = Stock('MSFT', start_date=datetime.date(2015, 1, 1), end_date=datetime.date(2021, 1, 1))
    df = pd.DataFrame(MSFT.stock_dict)
    df["date"] = pd.to_datetime(df["date"])
    output_file("candlestick.html")
    candlestick_plot(df)

