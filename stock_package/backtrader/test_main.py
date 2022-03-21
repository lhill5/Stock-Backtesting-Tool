import pandas as pd

def foo(x):
    x['c'] = 3

x = pd.DataFrame([{'a': 1, 'b': 2}])
print(x)

foo(x)
print(x)

