from bokeh.events import Pan, PanStart, PanEnd
from bokeh.models import ColumnDataSource, CustomJS, Rect, Range1d, Button
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import row, column


output_file("box_select_tool_callback.html")

draw_rect_source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[]))
draw_lines_source = ColumnDataSource(data=dict(x=[], y=[]))
button_source = ColumnDataSource(data=dict(value=[True]))

y_limits = Range1d(start=0, end=1)
p = figure(plot_width=400, plot_height=400, tools="pan",
           title="Select Below", x_range=(0, 1), y_range=y_limits)

rect = Rect(x='x', y='y', width='width', height='height',
            fill_alpha=0.3, fill_color='#009933')

p.add_glyph(draw_rect_source, rect, selection_glyph=rect, nonselection_glyph=rect)
p.multi_line('x', 'y', line_width=2, source=draw_lines_source)

draw_line_callback = CustomJS(args=dict(source=draw_lines_source, button_source=button_source), code="""
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

draw_rect_callback = CustomJS(args=dict(source=draw_rect_source, button_source=button_source, y_limits=(y_limits.start, y_limits.end)), code="""
    const data = source.data
    const y_min = y_limits[0]
    const y_max = y_limits[1]
    const button_value = button_source.data['value'][0]
    
    if (button_value) {
        
        let x, y, width, height
        height = y_max - y_min
        y = height/2
        
        let event = cb_obj.event_name
        console.log(event)
        
        if (event == 'panstart') {
            width = 0
            x = cb_obj['x'] + (width / 2)
            localStorage.setItem('line_x0', x)
        }
        else if (event == 'pan') {
            let x0 = Number(localStorage.getItem('line_x0'))
            width = cb_obj['x'] - x0
            x = x0 + (width / 2)
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
        
        // emit update of data source
        source.change.emit()
    }
""")

events = [Pan, PanStart, PanEnd]
for event in events:
    p.js_on_event(event, draw_rect_callback)

point_events = [PanStart, PanEnd]
for event in point_events:
    p.js_on_event(event, draw_line_callback)

p.js_on_event(Pan, draw_line_callback)

button = Button(label="GFG")
button.js_on_click(
    CustomJS(args=dict(source=button_source), code="""
        const data = source.data

        data['value'][0] = !data['value'][0]

        console.log(data['value'][0])
        source.change.emit()            
    """)
)

layout = column(button, p)
show(layout)

