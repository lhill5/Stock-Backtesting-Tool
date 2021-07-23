from bokeh.events import SelectionGeometry
from bokeh.models import ColumnDataSource, CustomJS, Rect, Range1d
from bokeh.plotting import figure, output_file, show

output_file("box_select_tool_callback.html")

source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[]))

y_limits = Range1d(start=0, end=1)
p = figure(plot_width=400, plot_height=400, tools="box_select",
           title="Select Below", x_range=(0, 1), y_range=y_limits)

rect = Rect(x='x', y='y', width='width', height='height',
            fill_alpha=0.3, fill_color='#009933')

p.add_glyph(source, rect, selection_glyph=rect, nonselection_glyph=rect)


callback = CustomJS(args=dict(source=source, y_limits=(y_limits.start, y_limits.end)), code="""

    const geometry = cb_obj['geometry']
    const data = source.data
    const y_min = y_limits[0];
    const y_max = y_limits[1];

    // calculate Rect attributes
    const width = geometry['x1'] - geometry['x0']
    const height = y_max - y_min;
    const x = geometry['x0'] + width/2
    const y = height/2

    // update data source with new Rect attributes
    data['x'].pop();
    data['y'].pop();
    data['width'].pop();
    data['height'].pop();
    
    data['x'].push(x)
    data['y'].push(y);
    data['width'].push(width);
    data['height'].push(height)

    // emit update of data source
    source.change.emit()
""")
p.js_on_event(SelectionGeometry, callback)

show(p)

