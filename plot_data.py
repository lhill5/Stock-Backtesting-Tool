import matplotlib.pyplot as plt
import mplfinance as mpf

from filter_data import filter, filter_year
from matplotlib import dates as dt
from datetime import timedelta
import pandas as pd
import numpy as np
import math


class Plot:
	def __init__(self, stock):
		self.stock = stock
		self.stock_fig = None
		self.stock_ax = None
		self.MACD_ax = None
		self.RSI_ax = None

		self.hover_lines = {}
		self.annotate_boxes = {}
		self.mouse_on_graph = False
		self.start, end = None, None
		self.start_index, end_index = None, None
		self.min_index, max_index = None, None
		self.last_date_hovered = False
		self.start, self.end, self.start_index, self.end_index = None, None, None, None

		self.arr_size = len(self.stock.prices['close'])
		self.next_plot = False
		self.plot_prices, self.plot_MACD, self.plot_RSI, self.plot_EMA = True, True, True, True

		self.dates, self.prices = self.stock.dates, self.stock.prices['close']
		self.MACD, self.signal, self.histogram = self.stock.tech_indicators['MACD'], self.stock.tech_indicators['signal'], self.stock.tech_indicators['histogram']
		self.EMA = self.stock.tech_indicators['EMA']
		self.SMA = self.stock.tech_indicators['SMA']
		self.RSI = self.stock.tech_indicators['RSI']


	def plot_data(self, days=0, months=0, years=0, year_num=0):

		# MACD_signals, EMA_signals, RSI_signals = self.stock.MACD_signals, self.stock.EMA_signals, self.stock.RSI_signals

		# filter stock data by date range
		# if year_num != 0:
		# 	dates, stock_data = filter_year(dates, prices, MACD, signal, EMA[50], year=year_num)
		# else:
		# 	dates, stock_data = filter(dates, prices, MACD, signal, EMA[50], days=days, months=months, years=years)
		# prices, MACD, signal, EMA_50 = stock_data

		# styling stuff
		plt.style.use("seaborn-dark")
		for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
			plt.rcParams[param] = '#212946'  # bluish dark grey
		for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
			plt.rcParams[param] = '0.9'  # very light grey

		stock_fig = plt.figure(figsize=(12, 8))
		plt.ion()

		stock_fig.canvas.mpl_connect('motion_notify_event', self.hover)
		stock_fig.canvas.mpl_connect('button_press_event', self.click)
		stock_fig.canvas.mpl_connect('button_release_event', self.click_release)
		stock_fig.canvas.mpl_connect('key_press_event', self.keypress)
		self.stock_fig = stock_fig

		stock_rows, MACD_rows, RSI_rows = get_plot_heights(self.plot_prices, self.plot_MACD, self.plot_RSI)
		# ind = [i for i in range(self.arr_size)] # the evenly spaced plot indices
		ind = [i for i in range(len(self.dates))]

		stock_ax, MACD_ax, RSI_ax = None, None, None
		if self.plot_prices:
			stock_ax = plt.subplot2grid((20, 1), (0, 0), rowspan=stock_rows)  # rows/columns/fig_number
			stock_ax.set_title(self.stock.ticker)

			data = {'Date': self.dates, 'Open': self.stock.prices['open'], 'High': self.stock.prices['high'], 'Low': self.stock.prices['low'], 'Close': self.stock.prices['close']}
			print(self.stock.ticker)
			df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close'])
			df.index = pd.DatetimeIndex(df['Date'])

			ap0 = [mpf.make_addplot(self.EMA[9], color='#F88017', ax=stock_ax),  # uses panel 0 by default
				   mpf.make_addplot(self.EMA[50], color='#0041C2', ax=stock_ax),  # uses panel 0 by default
				   mpf.make_addplot(self.EMA[200], color='#3CB043', ax=stock_ax)
			]
			mc = mpf.make_marketcolors(up='#0DB586', down='r', inherit=True)
			s = mpf.make_mpf_style(marketcolors=mc)
			if self.plot_EMA:
				mpf.plot(df, type='candle', style=s, ax=stock_ax, addplot=ap0)
			else:
				mpf.plot(df, type='candle', style=s, ax=stock_ax)

			stock_ax.grid(color='#2A3459')  # bluish dark grey, but slightly lighter than background
			stock_ax.xaxis.set_major_formatter(self.format_date)
			stock_ax.set_ylabel('Stock Price')
			stock_ax.set_xticks([])

			# self.annotate_trade_signals(stock_ax, EMA_signals)

		if self.plot_MACD:
			MACD_ax = plt.subplot2grid((20, 1), (stock_rows, 0), rowspan=MACD_rows)
			MACD_ax.plot(ind, self.MACD, '#fa9933')
			MACD_ax.plot(ind, self.signal, '#5985ff')

			green_histogram = [val if val is not None and val > 0 else 0 for val in self.histogram]
			red_histogram = [val if val is not None and val < 0 else 0 for val in self.histogram]

			MACD_ax.bar(ind, green_histogram, color='green')
			MACD_ax.bar(ind, red_histogram, color='red')

			MACD_ax.grid(color='#2A3459')  # bluish dark grey, but slightly lighter than background
			MACD_ax.xaxis.set_major_formatter(self.format_date)

			MACD_ax.set_ylabel('MACD/signal')
			MACD_ax.set_xlabel('Timestamp')

			# self.annotate_trade_signals(MACD_ax, MACD_signals)

		if self.plot_RSI:
			RSI_ax = plt.subplot2grid((20, 1), (stock_rows + MACD_rows, 0), rowspan=RSI_rows)
			RSI_ax.plot(ind, self.RSI, '#5985ff')
			RSI_ax.grid(color='#2A3459')  # bluish dark grey, but slightly lighter than background

			# 	# plot horizontal 30, 70 line thresholds
			RSI_ax.axhline(y=30, color='#C11B17')
			RSI_ax.axhline(y=70, color='#C11B17')

			RSI_ax.xaxis.set_major_formatter(self.format_date)

			RSI_ax.set_ylabel('RSI Value')
			RSI_ax.set_xlabel('timestamp')

			# self.annotate_trade_signals(RSI_ax, RSI_signals)

		stock_fig.autofmt_xdate()

		# print(f'EMA: {len(EMA_signals)}, MACD: {len(MACD_signals)}, RSI: {len(RSI_signals)}')

		self.stock_fig = stock_fig
		self.stock_ax, self.MACD_ax, self.RSI_ax = stock_ax, MACD_ax, RSI_ax
		plt.show(block=True)


	def annotate_chart(self, ax, ax_name, date, value):

		if ax_name in self.annotate_boxes and self.annotate_boxes[ax_name] is not None:
			try:
				self.annotate_boxes[ax_name].remove()
			except:
				pass

		arrowprops = dict(arrowstyle="wedge,tail_width=0.5", alpha=0.7, color='b')
		bbox = dict(boxstyle="round", alpha=0.7, color='b')

		y_min, y_max = ax.get_ylim()
		self.annotate_boxes[ax_name] = ax.annotate(
			value,
			xy=(date, (0.075 * (y_max - y_min)) + value),
			xytext=(0, 0),
			textcoords='offset points',
			size=13,
			ha='center', va="center",
			bbox=bbox,
			arrowprops=arrowprops
		)


	def draw_buy_signal(self, ax, date, y_val):
		arrowprops = dict(arrowstyle="simple", alpha=0.7, color='g')
		bbox = dict(boxstyle="round", alpha=0.7, color='g')

		x_val = self.stock.date_to_index[date]
		y_min, y_max = ax.get_ylim()
		ax.annotate(
			'buy',
			xy=(x_val, y_val),
			xytext=(0, -25),  # (1 * (y_max - y_min))),
			textcoords='offset points',
			size=12,
			ha='center', va="center",
			bbox=bbox,
			arrowprops=arrowprops
		)


	def draw_sell_signal(self, ax, date, y_val):
		arrowprops = dict(arrowstyle="simple", alpha=0.7, color='r')
		bbox = dict(boxstyle="round", alpha=0.7, color='r')

		x_val = self.stock.date_to_index[date]
		y_min, y_max = ax.get_ylim()
		ax.annotate(
			'sell',
			xy=(x_val, y_val),
			xytext=(0, 25),  # (1 * (y_max - y_min))),
			textcoords='offset points',
			size=12,
			ha='center', va="center",
			bbox=bbox,
			arrowprops=arrowprops
		)


	def annotate_trade_signals(self, ax, signals):

		for signal in signals:
			date, price, action = signal
			index = self.stock.date_to_index[date]
			if ax is self.MACD_ax:
				y_val = self.MACD[index]
			elif ax is self.RSI_ax:
				y_val = self.RSI[index]
			else:
				y_val = price

			# print(action, date, y_val)
			if action == 'buy':
				self.draw_buy_signal(ax, date, y_val)
			else:
				assert (action == 'sell')
				self.draw_sell_signal(ax, date, y_val)


	def draw_hover_line(self, ax, ax_name, x_val):

		if ax_name in self.hover_lines and self.hover_lines[ax_name] in ax.lines:
			l = ax.lines.remove(self.hover_lines[ax_name])
			del l

		self.hover_lines[ax_name] = ax.axvline(x=x_val, color='#5353ff')


	def hover(self, event):

		if event.xdata is None:
			return
		# print(round(event.xdata))
		index = min(round(event.xdata), len(self.stock.unique_indices) - 1)
		date = (self.stock.index_to_date[index])
		# print(self.stock_ax)

		if date is not None:
			self.last_date_hovered = date

			price = self.stock.prices['close'][index]
			MACD_val = self.stock.tech_indicators['MACD'][index]
			RSI_val = self.stock.tech_indicators['RSI'][index]

			if price is not None:
				if self.stock_ax:
					self.annotate_chart(self.stock_ax, 'stock_ax', round(event.xdata), round(price,2))
					self.draw_hover_line(self.stock_ax, 'stock_ax', event.xdata)
				if self.MACD_ax:
					self.annotate_chart(self.MACD_ax, 'MACD_ax', round(event.xdata), round(MACD_val,2))
					self.draw_hover_line(self.MACD_ax, 'MACD_ax', event.xdata)
				if self.RSI_ax:
					self.annotate_chart(self.RSI_ax, 'RSI_ax', round(event.xdata), round(RSI_val,2))
					self.draw_hover_line(self.RSI_ax, 'RSI_ax', event.xdata)


	def click(self, event):

		if event.xdata is None:
			return
		self.start_index = round(event.xdata)
		self.start = self.stock.index_to_date[round(event.xdata)]


	def click_release(self, event):

		start, end, start_index = self.start, self.end, self.start_index
		# if user released click off the graph (left or right)
		if event.xdata is None:
			if self.last_date_hovered is not None:
				# use last hovered date to determine if hovered off left or right of graph
				if self.last_date_hovered < start:
					end = self.stock.start_date
				else:
					end = self.stock.end_date
			else:
				end = self.stock.max_date
			end_index = self.stock.date_to_index[end]
		else:
			# index = round(event.xdata)
			# end = stock_dict[current_ticker]['dates'][index]
			end_index = round(event.xdata)
			end = self.stock.index_to_date[end_index]

		# reverse start/end dates if user dragged from right to left
		if start > end:
			start, end = end, start
			start_index, end_index = end_index, start_index

		# start = self.stock.date_to_index[start]
		# end = self.stock.date_to_index[start]

		if self.stock_ax:
			self.stock_ax.set_xlim(start_index, end_index)
			self.set_y_lim(self.stock_ax, self.prices, start_index, end_index)

		if self.MACD_ax:
			self.MACD_ax.set_xlim(start_index, end_index)
			self.set_y_lim(self.MACD_ax, self.MACD, start_index, end_index)

		if self.RSI_ax:
			self.RSI_ax.set_xlim(start_index, end_index)
			self.set_y_lim(self.RSI_ax, self.RSI, start_index, end_index)

		self.start = self.end = self.start_index = self.end_index = None


	def keypress(self, event):
		global next_plot

		key_pressed = event.key.lower()
		if key_pressed == 'r':
			self.reset(self.stock_ax, self.prices)
			self.reset(self.MACD_ax, self.MACD)
			self.reset(self.RSI_ax, self.RSI)

		elif key_pressed == 'left':
			self.shift_left(self.stock_ax, self.prices)
			self.shift_left(self.MACD_ax, self.MACD)
			self.shift_left(self.RSI_ax, self.RSI)

		elif key_pressed == 'right':
			self.shift_right(self.stock_ax, self.prices)
			self.shift_right(self.MACD_ax, self.MACD)
			self.shift_right(self.RSI_ax, self.RSI)

		elif key_pressed == 'n':
			plt.close()
			self.next_plot = True


	def shift_left(self, ax, arr):
		start, end = self.start, self.end

		if ax:
			sub_days = 10
			x_min, x_max = map(convert_todate, ax.get_xlim())
			x_min -= timedelta(sub_days)
			x_max -= timedelta(sub_days)

			if self.is_inrange(x_min) and self.is_inrange(x_max) and x_min < x_max:
				ax.set_xlim(x_min, x_max)
				self.set_y_lim(ax, arr, x_min, x_max)


	def shift_right(self, ax, arr):
		start, end = self.start, self.end

		if ax:
			sub_days = 10
			x_min, x_max = map(convert_todate, ax.get_xlim())
			x_min += timedelta(sub_days)
			x_max += timedelta(sub_days)

			if self.is_inrange(x_min) and self.is_inrange(x_max) and x_min < x_max:

				ax.set_xlim(x_min, x_max)
				self.set_y_lim(ax, arr, x_min, x_max)


	def reset(self, ax, arr):
		min_date, max_date = self.stock.start_date, self.stock.end_date
		min_index, max_index = self.stock.date_to_index[min_date], self.stock.date_to_index[max_date]

		start_i = self.stock.date_to_index[min_date]
		end_i = self.stock.date_to_index[max_date]
		if ax:
			ax.set_xlim(0, end_i-start_i)
			self.set_y_lim(ax, arr, 0, end_i-start_i)


	def set_y_lim(self, ax, arr, start, end):
		global stock_ax, MACD_ax, RSI_ax

		min_price, max_price = self.get_min_max(start, end, arr)
		ax.set_ylim(min_price - ((max_price - min_price) * 0.075), max_price + ((max_price - min_price) * 0.075))


	def get_min_max(self, start, end, arr):

		min_val = min(arr[start: end+1])
		max_val = max(arr[start: end+1])

		return min_val, max_val


	def is_inrange(self, date):
		min_date = self.stock.dates[0]
		max_date = self.stock.dates[-1]
		return date >= min_date and date <= max_date


	def format_date(self, x, pos=None):
		thisind = np.clip(int(x + 0.5), 0, self.arr_size - 1)
		date = self.dates[thisind].strftime('%Y-%m-%d')
		return date


def get_plot_heights(plot_stock, plot_MACD, plot_RSI):
	if plot_stock and plot_MACD and plot_RSI:
		stock_rows = 10
		MACD_rows = RSI_rows = 5
	elif plot_stock and (plot_MACD or plot_RSI):
		stock_rows = 15
		MACD_rows = 5 if plot_MACD else 0
		RSI_rows = 5 if plot_RSI else 0
	else:
		stock_rows = 20
		MACD_rows = RSI_rows = 0

	return stock_rows, MACD_rows, RSI_rows


def convert_todate(xval):
	if xval is None:
		return None

	date = dt.num2date(xval).date()
	return date

