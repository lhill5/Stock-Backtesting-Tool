from bokeh.plotting import figure


# class for candlestick chart
class CandlestickChart:
    def __init(self, stock):
        self.stock = stock

        TOOLS = 'wheel_zoom, reset, save'
        self.len = len(stock.dates)
        self.start, self.end = 0, self.len - 1
        self.y_limits = self.get_starting_y_limits()
        self.fig = figure(title="stock prices", title_location='above', x_axis_type='linear', width=1400, height=400, toolbar_location="right", tools=TOOLS, x_range=(self.start, self.end), y_range=(self.y_limits[0], self.y_limits[1]))


    def get_starting_y_limits(self):
        start_i = self.start
        end_i = self.end

        y_min = min(self.stock.low[start_i:end_i+1])
        y_max = max(self.stock.high[start_i:end_i+1])

        # 5 percent offset on upper/lower limit of graph
        offset = 0.05
        return y_min - offset*y_min, y_max + offset*y_max
