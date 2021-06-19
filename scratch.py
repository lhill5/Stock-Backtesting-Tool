import pandas as pd

df = pd.read_csv('stock_data/AAPL.csv', index_col=0)

cols = pd.MultiIndex.from_tuples([("Open",), ("High",), ("Low",), ("Close",), ("Adj Close",), ("Volume",)])

df.columns = cols
df = df.to_dict()
df = {key[0].lower(): value for key, value in df.items()}

print(df)

