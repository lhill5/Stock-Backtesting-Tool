import pandas as pd
import mplfinance as mpf
from datetime import datetime
import matplotlib.pyplot as plt

# Load data file.
# {'Date': dates, 'Open': _open, 'High': high, 'Low': low, 'Close': close}
dates = [datetime(2020,1,i) for i in range(1, 10)]
data = {'Date': dates,
        'Open': [ 0.1,  0.5, 1.3, 5, 3, 1,   1.5, 2,   5],
        'High': [ 0.2,  0.5, 1.8, 7, 3, 2,   3.2, 2,   6],
        'Low': [  0.05, 0.2, 1,   3, 2, 1,   1.5, 1.7, 4.3],
        'Close': [0.05, 0.3, 1.8, 4, 3, 1.5, 3.2, 1.8, 4.3]
        }

df = pd.DataFrame(data, columns = ['Date','Open','High','Low','Close'], index=dates)


# Plot candlestick.
# Add volume.
# Add moving averages: 3,6,9.
# Save graph to *.png.
print(type(dates), type(dates[0]), type(df), df.index)

mpf.plot(df, type='candle', style='charles',
         title='S&P 500, Nov 2019',
         ylabel='Price ($)',
         ylabel_lower='Shares \nTraded')

plt.show()
