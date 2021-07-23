from bokeh.models import Button, CustomJS, ColumnDataSource
from bokeh.plotting import show


source = ColumnDataSource(data=dict(var=[0]))

button = Button(label="GFG")
button.js_on_click(
    CustomJS(args=dict(source=source), code="""
        const data = source.data
        
        data['var'][0] = !data['var'][0]
    
        console.log(data['var'][0])
        source.change.emit()            
    """)
)

show(button)

print(source.data['var'])

