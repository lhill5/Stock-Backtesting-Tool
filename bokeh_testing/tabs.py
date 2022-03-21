from bokeh.layouts import Column, Row
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.plotting import figure, curdoc, show
from bokeh.models.widgets import Button, Tabs, Panel
from bokeh.events import Tap
import numpy as np

tabs_array = []
data = [dict(x = np.arange(10), y = np.random.random(10)) for counter in range(1, 5)]
data_sources = [ColumnDataSource(dict(x = np.arange(10), y = np.random.random(10))) for counter in range(1, 3)]

def button_callback():
    for index in range(2):
        data_sources[index].data = data[index + 2]

button = Button(label = 'Swap DataSource', width = 100)

def get_plots():
    row = Row()
    plots = [figure(plot_height = 400, tools = '') for i in range(2)]
    [plot.line(x = 'x', y = 'y', source = data_sources[index]) for index, plot in enumerate(plots)]
    [row.children.append(plot) for plot in plots]
    return row

def make_tab(tab_nmb):
    if tab_nmb:
        return Panel(title = '{name}'.format(name = tab_nmb), child = button)
    else:
        return Panel(title = '{name}'.format(name = tab_nmb), child = get_plots())

[tabs_array.append(make_tab(tab_nmb)) for tab_nmb in range(2)]
tabs = Tabs(tabs = tabs_array)

def get_callback(val):
    return CustomJS(args = dict(val = val, tabs = tabs), code = """
    if (val < 0)
        tabs.active = tabs.active + val > -1 ? tabs.active + val : tabs.tabs.length -1;
    if (val > 0)
        tabs.active = tabs.active + val < tabs.tabs.length ? tabs.active + val : 0;""")

button.js_on_event(Tap, get_callback(-1))
button.on_click(button_callback)

document = curdoc()
document.add_root(tabs)
show(tabs)