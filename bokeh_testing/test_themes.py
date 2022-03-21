# mybokeh_themes.py

from bokeh.plotting import figure, curdoc
from bokeh.palettes import Spectral
from bokeh.themes import Theme
from bokeh.document import Document

# import some layout elements, a select widget, and the built_in_themes
from bokeh.layouts import column
from bokeh.models import Select
from bokeh.themes import Theme, built_in_themes

def switch_theme(value, old, new):
    curdoc().theme = new


x_f = [1.5, 2, 9]
y_f = [3, 3, 3.1]

p = figure(plot_width=400, plot_height=400)
p.line(x_f, y_f, line_width=3, color=Spectral[4][0])


theme_select = Select(title='Theme', options=['caliber',
                                              'dark_minimal',
                                              'light_minimal',
                                              'night_sky',
                                              'contrast'])
theme_select.on_change('value', switch_theme)

curdoc().add_root(column(theme_select, p))