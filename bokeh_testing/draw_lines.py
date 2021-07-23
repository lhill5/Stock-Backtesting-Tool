from bokeh import events
from bokeh.io import show
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.plotting import figure


def draw_line_event():
    "Build a suitable CustomJS to display the current event in the div model."
    return CustomJS(args=dict(source=source), code="""
        const data = source.data;
        console.log(cb_obj.event_name);
        
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
    """)


source = ColumnDataSource(data=dict(x=[], y=[]))
p = figure(plot_width=400, plot_height=400, tools="",
           title="Select Below", x_range=(0, 1), y_range=(0, 1))
p.multi_line('x', 'y', legend_label="Temp.", line_width=2, source=source)


## Events with attributes
point_attributes = ['x', 'y', 'sx', 'sy']
pan_attributes = point_attributes + ['delta_x', 'delta_y'] # Pan event

point_events = [events.PanStart, events.PanEnd]

for event in point_events:
    p.js_on_event(event, draw_line_event())

p.js_on_event(events.Pan, draw_line_event())

# output_file("js_events.html", title="JS Events Example")
show(p)

