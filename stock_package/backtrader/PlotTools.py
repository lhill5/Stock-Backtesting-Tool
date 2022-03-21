import math
from bokeh.plotting import figure
from CandlestickChart import CandlestickChart
from bokeh.models import CrosshairTool, Panel, Tabs
from bokeh.layouts import row, column


"""
so, profit loss ranges plots for a figure, the ranges
need some way to select between visibility of tab's candlestick chart


"""
class BokehTab:
    profit_loss_ranges = []
    buysell_scatter_charts = []
    tabs_layout = Tabs(tabs=[])
    prev_tab = None

    def __init__(self, graph, name, start_date_slicer=None, end_date_slicer=None, select_stock=None, title=None):
        self.graph = graph
        self.name = name
        self.start_widget = start_date_slicer
        self.end_widget = end_date_slicer
        self.stock_widget = select_stock

        if self.tabs_layout.tabs == []:
            self.tabs_layout.on_change('active', self.change_tab)

        self.tab = None

        # self.change_tab(None, None, 0)  # displays buysell lines for first tab initially (index 0)


    def add(self, *figs, add_widgets=True):
        if len(figs) == 0:
            return

        if add_widgets:
            # assumes tabs with widgets also will plot ranges where user should've bought/sold stock
            self.plot_range(figs[0])
            try:
                tab = Panel(child=column(row(self.start_widget, self.end_widget, self.stock_widget), *figs), title=f'{self.name} chart')
            except:
                print('error creating tab, exiting ...')
                exit(0)
        else:
            self.profit_loss_ranges.append(None)
            tab = Panel(child=column(*figs), title=f'{self.name} chart')

        self.tab = tab
        self.tabs_layout.tabs.append(self.tab)


    # graphs buy/sell ranges and keeps track of buy/sell glyphs
    def plot_range(self, first_chart):
        # 1st chart is where we add buy/sell ranges to
        # by default glyphs are not visible
        glyph_boxes = self.graph.plot_profit_loss_ranges(first_chart, self.name)
        self.profit_loss_ranges.append(glyph_boxes)
        if len(self.profit_loss_ranges) == 1:
            self.change_tab(None, None, 0)


    def change_tab(self, attr, old, new):
        active_tab = new
        valid_tab_len = len(self.profit_loss_ranges)-1
        # print(f'prev tab: {self.prev_tab}, active tab: {active_tab}')

        # if user clicks on a non tech indicator chart, like fundamental data, then do nothing
        if active_tab and active_tab > valid_tab_len:
            return

        # can occur if user clicks tab outside of valid tab len then back to prev tab
        if active_tab == self.prev_tab:
            return

        if self.prev_tab is not None:
            if self.profit_loss_ranges[self.prev_tab] is not None:
                for profit_loss_box in self.profit_loss_ranges[self.prev_tab]:
                    profit_loss_box.visible = False

            # for profit_loss_box in self.buysell_scatter_charts[self.prev_tab]:
            #     profit_loss_box.visible = False

        # breakpoint()
        if self.profit_loss_ranges[active_tab] is not None:
            for profit_loss_box in self.profit_loss_ranges[active_tab]:
                profit_loss_box.visible = True

        self.prev_tab = active_tab


# handles easily creating new figures in graph library
class SubPlot:
    def __init__(self, name, data, plot_func, candlestick_fig):
        self.name = name
        self.data = data
        self.candlestick_fig = candlestick_fig
        self.fig = figure(x_axis_type='linear', tools="crosshair", toolbar_location=None, width=1400, height=200, x_range=candlestick_fig.x_range)

        crosshair = CrosshairTool(
            dimensions='height',
            line_color="#cbcbcb",
            line_width=0.6
        )
        self.fig.add_tools(crosshair)

        # init autoscaling for figure
        self.fig.x_range.on_change('start', self.autoscale_yaxis)
        self.fig.x_range.on_change('end', self.autoscale_yaxis)

        # plots data to a figure in bokeh
        plot_func(self.fig)
        self.autoscale_yaxis(1,1,1)
        # graph.plot_buysell_lines(self.candlestick_fig, name)
        # buy_chart, sell_chart = graph.plot_buysell_scatter_chart(self.candlestick_fig, name)
        # BokehTab.buysell_scatter_charts.append([buy_chart, sell_chart])


    def autoscale_yaxis(self, attr, old, new):
        if isinstance(self.data, dict):
            index = [i for i in range(len(list(self.data.values())[0]))]
        else:
            index = [i for i in range(len(self.data))]

        x_range = self.fig.x_range
        y_range = self.fig.y_range
        start, end = x_range.start, x_range.end

        min_val = math.inf
        max_val = -math.inf

        for i in index:
            if i >= start and i <= end:
                if (isinstance(self.data, dict)):
                    for key in self.data:
                        max_val = max(max_val, self.data[key][i])
                        min_val = min(min_val, self.data[key][i])
                else:
                    max_val = max(max_val, self.data[i])
                    min_val = min(min_val, self.data[i])

        pad = (max_val - min_val) * 0.05
        y_range.start = min_val - pad
        y_range.end = max_val + pad


    def replot(self):
        self.autoscale_yaxis(1, 1, 1)
