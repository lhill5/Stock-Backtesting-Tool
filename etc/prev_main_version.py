# from __future__ import division

import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.dates import num2date
from matplotlib.ticker import FuncFormatter

import dateutil.parser
from matplotlib import cbook, dates as dt
from matplotlib.ticker import Formatter

from datetime import datetime, timedelta
import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import math
import tkinter as tk
from collections import defaultdict
import random
# need to get second data not daily data
import os, sys
import keyboard

write_mode = 0
print_to_ext_file = 0


stock_fig = None
stock_ax = None
MACD_ax = None
RSI_ax = None
stock_dict = None
hover_lines = {}
annotate_boxes = {}
mouse_on_graph = False
start, end = None, None
start_index, end_index = None, None
min_date, max_date = None, None
min_index, max_index = None, None
last_date_hovered = False
N = None

plot_stock = True
plot_MACD = True
plot_RSI = True

next_plot = False
plot_data = True

current_ticker = None

#
# class MyFormatter(Formatter):
# 	def __init__(self, dates, fmt='%Y-%m-%d'):
# 		self.dates = dates
# 		self.fmt = fmt
#
# 	def __call__(self, x, pos=0):
# 		"""Return the label for time x at position pos."""
# 		ind = int(round(x))
# 		if ind >= len(self.dates) or ind < 0:
# 			return ''
# 		return dt.num2date(self.dates[ind]).strftime(self.fmt)


def store_stock_data(stock_list=None):

	count, error_count = 0, 0
	stocks_dict = {}
	if stock_list is None:
		tickers = pd.read_csv('s&p_500.csv', names=['ticker symbols'])

		# i = 0
		for col, stock in tickers.iterrows():
			# i += 1
			# if i < 400 or i > 413:
			# 	continue

			ticker, name, *_ = stock[0].split(' - ')
			if '.' in ticker:
				continue

			print(ticker)
			try:
				data = stock_lookup(ticker)
				stocks_dict[ticker] = {'name':name, 'data': data}
			except KeyError:
				error_count += 1
			count += 1
			print(count, error_count)

		stocks_dict = pd.DataFrame.from_dict(stocks_dict)
		if os.path.exists('sp500_dictionary.pkl'):
			os.remove('sp500_dictionary.pkl')
		stocks_dict.to_pickle('sp500_dictionary.pkl')

	else:
		for ticker in stock_list:
			if '.' in ticker:
				continue

			name = ticker
			data = stock_lookup(ticker)
			stocks_dict[ticker] = {'name': name, 'data': data}

		stocks_dict = pd.DataFrame.from_dict(stocks_dict)
		if os.path.exists('stocklist_dictionary.pkl'):
			os.remove('stocklist_dictionary.pkl')
		stocks_dict.to_pickle('stocklist_dictionary.pkl')


def stock_lookup(stock):
	# ticker = yf.Ticker(stock)
	# info = ticker.info
	# history = ticker.history(period="max")

	yf.pdr_override()
	data = pdr.get_data_yahoo(stock)

	stock_dict = {}
	dates = []

	for key, value in data.items():
		dd = defaultdict(list)
		arr = value.to_dict(dd)
		prices = []
		for timestamp, price in arr.items():
			if not math.isnan(price):
				date = timestamp
				if key == 'Adj Close':
					dates.append(date)
				if price < 0:
					print(f'{stock} is negative :/')
					exit(1)
				prices.append(price)

		# key = 'prices' if key == 'Adj Close' else key
		if key == 'Close':
			stock_dict['prices'] = prices
		stock_dict[key.lower()] = prices

	stock_dict['dates'] = dates
	return stock_dict

def read_stored_stocks(stock_list=False):
	if stock_list:
		stock_dict = pd.read_pickle('stocklist_dictionary.pkl')
	else:
		stock_dict = pd.read_pickle('sp500_dictionary.pkl')

	return stock_dict.to_dict()


def get_MACD(dates, prices):
	fast_EMA_days = 12
	slow_EMA_days = 26
	difference = 0
	signal_days = 9

	fast_EMA_factor = 2 / (fast_EMA_days+1)
	slow_EMA_factor = 2 / (slow_EMA_days+1)
	signal_factor = 2 / (signal_days+1)

	fast_EMA = [sum(prices[:9]) / 9]
	prev_val = fast_EMA[0]
	for i, val in enumerate(prices[9:]):
		# if math.isnan(val):
		# 	cur_val = prev_val
		# else:
		cur_val = ((val - prev_val) * fast_EMA_factor) + prev_val
		fast_EMA.append(cur_val)
		prev_val = cur_val

	slow_EMA = [sum(prices[:26]) / 26]
	prev_val = slow_EMA[0]
	for i, val in enumerate(prices[26:]):
		# if math.isnan(val):
		# 	cur_val = prev_val
		# else:
		cur_val = ((val - prev_val) * slow_EMA_factor) + prev_val
		slow_EMA.append(cur_val)
		prev_val = cur_val

	difference = [fast - slow for fast, slow in zip(fast_EMA[17:], slow_EMA)]
	assert(len(fast_EMA[17:]) == len(slow_EMA))

	signal = [sum(difference[:9]) / 9]
	for i, val in enumerate(difference[9:]):
		prev_val = signal[i]
		cur_val = ((val - prev_val) * signal_factor) + prev_val
		signal.append(cur_val)

	histogram = [dif - sig for dif, sig in zip(difference[8:], signal)]

	difference = difference[8:]

	left_fill = [np.nan for _ in range(33)]
	return left_fill + difference, left_fill + signal, left_fill + histogram


def get_EMA(dates, prices, num_days):
	factor = 2 / (num_days+1)
	EMA = [sum(prices[:num_days]) / num_days]

	prev_val = EMA[0]
	for date, val in zip(dates[num_days:], prices[num_days:]):
		# if math.isnan(val):
		# 	cur_val = prev_val
		# else:
		cur_val = ((val - prev_val) * factor) + prev_val
		EMA.append(cur_val)
		prev_val = cur_val

	left_fill = [np.nan for _ in range(num_days-1)]
	return left_fill + EMA


def get_RSI(dates, prices):

	RSI = []
	RSI_period = 14
	prev_close = prices[0]
	gains, losses = [], []

	for close_price in prices[1:RSI_period+1]:
		change = close_price - prev_close
		if change > 0:
			gains.append(change)
			losses.append(0)
		else:
			losses.append(change)
			gains.append(0)
		prev_close = close_price

	average_gains = sum(gains) / len(gains)
	average_losses = abs(sum(losses) / len(losses))
	relative_strength = average_gains / average_losses if average_losses != 0 else 0
	RSI_step1 = 100 - (100 / (1 + relative_strength))

	RSI = [np.nan for _ in range(15)]
	RSI.append(RSI_step1) # first RSI value

	prev_avg_gain = average_gains
	prev_avg_loss = average_losses

	i = 0
	prev_close = prices[15]
	prev_RSI = RSI_step1
	for date, close in zip(dates[16:], prices[16:]):
		change = close - prev_close
		if change > 0:
			gain = change
			loss = 0
		else:
			loss = abs(change)
			gain = 0

		avg_up_movement = ((prev_avg_gain * (RSI_period-1)) + gain) / RSI_period
		avg_down_movement = ((prev_avg_loss * (RSI_period-1)) + loss) / RSI_period
		relative_strength = avg_up_movement / avg_down_movement if avg_down_movement != 0 else prev_RSI
		RSI_step2 = 100 - (100 / (1 + relative_strength))

		prev_close = close
		prev_avg_gain = avg_up_movement
		prev_avg_loss = avg_down_movement
		prev_RSI = RSI_step2
		RSI.append(RSI_step2)

	return RSI


def backtest_MACD(dates, prices, MACD, signal, histogram, EMA_arr):
	MACD_signals = []
	prev_MACD, prev_signal = MACD[0], signal[0]
	prev_hist = histogram[0]

	i = 1
	bought = False
	for date, price, MACD_val, signal_val, hist_val in zip(dates[1:], prices[1:], MACD[1:], signal[1:], histogram[1:]):
		action = None
		rate_of_change = (MACD_val - prev_MACD)
		RSI_val = get_indicator(date, 'RSI')

		# buy signal
		if i-2 >= 0 and histogram[i-2] < 0 and histogram[i-1] > 0 and histogram[i] > histogram[i-1]:
			rate_of_change = (histogram[i] - histogram[i-1]) / histogram[i-1]
			if RSI_val < 70: #histogram[i] > 0.2 and rate_of_change > 0.5:
			# if MACD_val < 0: #and rate_of_change >= 5:
				if not bought: #and rate_of_change >= 0.2:
					action = (date, price, "buy")
					bought = True
		# sell signal
		elif i-1 >= 0 and hist_val < 0:
			# "ignore sell signal if MACD stock indicator is above zero"
			if RSI_val > 30: #MACD_val > 0:
				if bought:
					action = (date, price, "sell")
					bought = False

		if action is not None:
			MACD_signals.append(action)

		prev_MACD = MACD_val
		prev_signal = signal_val
		prev_hist = hist_val

		i += 1

	# success_rate = 0 if total_trades == 0 else (successful_trades / total_trades) * 100
	return MACD_signals


def backtest_EMA(dates, prices, EMAs):
	if len(EMAs) < 2:
		return []

	EMA_signals = []
	bought = False

	for i in range(len(dates)):
		action = None

		# diffs = (EMAs[0][i] - EMAs[-1][i]) / prices[i]

		min_last_month = min(EMAs[0][i-365:i]) if i-365 >= 0 else math.inf

		if np.nan not in EMAs:
			# if not bought and all(EMA1[i] > EMA2[i] for EMA1, EMA2 in zip(EMAs, EMAs[1:])):
			rate_of_change = ((EMAs[0][i] - EMAs[-1][i]) / EMAs[-1][i]) * 100
			if not bought and rate_of_change >= 5 and EMAs[-2][i] > EMAs[-1][i]: #all(EMA1[i] > EMA2[i] for EMA1, EMA2 in zip(EMAs, EMAs[1:])):
				action = (dates[i], prices[i], 'buy')
				bought = True
			elif bought and EMAs[-2][i] < EMAs[-1][i]:
				# if EMAs[0][i] < min_last_month:
				action = (dates[i], prices[i], 'sell')
				bought = False

		if action is not None:
			EMA_signals.append(action)

	return EMA_signals


def backtest_RSI(dates, prices, RSI):

	RSI_signals = []
	bought = False
	for i, date in enumerate(dates):
		action = None

		if RSI[i] is np.nan:
			continue
		# buy signal
		elif RSI[i] < 30 and (i-1) >= 0 and RSI[i-1] >= 30:
			if not bought:
				action = (dates[i], round(prices[i], 2), 'buy')
				bought = True
		# sell signal
		elif RSI[i] >= 70 and (i-1) >= 0 and RSI[i-1] < 70:

			if bought:
				action = (dates[i], round(prices[i], 2), 'sell')
				bought = False

		if action is not None:
			RSI_signals.append(action)

	return RSI_signals



def filter_trade_signals(trade_signals, years=0, year_num=0, start_date=datetime.now(), end_date=datetime.now()):
	if len(trade_signals) < 2:
		return []

	cur_year = datetime.now().year
	start_date, end_date = start_date.date(), end_date.date()

	output = []
	for trade in trade_signals:
		date = trade[0]
		year = date.year
		if years != 0:
			if year >= cur_year-years:
				output.append(trade)
		elif year_num != 0 and str(year_num) in date:
			output.append(trade)
		elif date >= start_date and date <= end_date:
			output.append(trade)

	if len(output) <= 1:
		return []

	return output[1:] if output[0][2] == 'sell' else output


def filter_between(dates, *data, start_date, end_date):
	start, end = 0,0
	start_date, end_date = start_date.date(), end_date.date()
	for i, date in enumerate(dates):
		if date >= start_date and start == 0:
			start = i
		elif date <= end_date:
			end = i

	filtered_list = [d[start:end+1] for d in data]
	if len(filtered_list) == 1:
		filtered_list = filtered_list[0]

	return dates[start:end+1], filtered_list


def filter_year(dates, *data, year=2021):
	start, end = 0, 0
	for i, date in enumerate(dates):
		cur_year = date.year
		if date.year == year and start == 0:
			start = i
		elif cur_year > year and end == 0:
			end = i-1

	if end == 0:
		end = len(dates)-1

	filtered_list = [d[start:end+1] for d in data]
	if len(filtered_list) == 1:
		filtered_list = filtered_list[0]

	return dates[start:end+1], filtered_list


def filter(dates, *data, days=0, months=0, years=0):
	total_days = len(dates)

	if days != 0:
		num_days = days
	elif months != 0:
		num_days = months * 30
	elif years != 0:
		num_days = years * 365
	else:
		num_days = total_days+1 # defaults to return all data

	start = (total_days - num_days) + 1
	filtered_list = [d[start:] for d in data]
	if len(filtered_list) == 1:
		filtered_list = filtered_list[0]

	return dates[start:], filtered_list


def rank_trade_signals(trade_signals, min_date, max_date):
	if trade_signals is None or len(trade_signals) < 2:
		return [], None, None

	total_profit = 0
	successful_trades, total_trades = 0, 0
	bought,sold = 0,0
	num_trades = len(trade_signals) // 2

	prev_trade = None
	ROIs = []
	for trade in trade_signals:
		date, price, action = trade
		if action == 'buy':
			bought += price
			prev_trade = trade

		if action == 'sell' and prev_trade:
			sold += price
			ROI = (price - prev_trade[1]) / prev_trade[1]
			ROIs.append(ROI)

			profit = price - prev_trade[1]
			total_profit += profit
			if profit > 0:
				successful_trades += 1
			total_trades += 1

			# f.write(f'bought on {prev_trade[0]}, sold on {date}, profit {round(price - prev_trade[1], 2)}, ROI: {ROI}%\n')
			prev_trade = None

	avg_ROI = sum(ROIs) / len(ROIs) if len(ROIs) != 0 else 0
	gain_per_dollar = pow((1 + avg_ROI), num_trades)
	P = 1
	G = gain_per_dollar - P
	n = (max_date - min_date).days
	annualize_ROI = pow(((P + G) / P), (365 / n)) - 1
	annualize_ROI = round(annualize_ROI * 100, 2)

	success_rate = 0 if total_trades == 0 else round((successful_trades / total_trades) * 100, 1)
	total_profit = round(total_profit, 2)
	return total_profit, success_rate, annualize_ROI


def print_trade_results(total_profit, success_rate, avg_rate_of_return, num_trades, strat):
	if num_trades is not None:
		print(f'{strat} number of trades = {math.floor(num_trades)}')
	# if success_rate is not None:
	# 	print(f'{strat} success rate = {success_rate}%')
	if avg_rate_of_return is not None:
		print(f'{strat} annualized performance = {avg_rate_of_return}%')


def print_top_x(AROR, strat_name):
	print(f'\nbest {strat_name} ARORs:')
	for ticker, ror, buy_n_hold in AROR:
		print(ticker, f'{ror}%, buy and hold {buy_n_hold}')


def print_bottom_x(AROR, strat_name):
	print(f'\nworst {strat_name} ARORs:')
	for ticker, ror, buy_n_hold in AROR:
		print(ticker, f'{ror}%, buy and hold {buy_n_hold}')


def get_slope(x1, y1, x2, y2):
	if x2-x1 == 0:
		return 100
	else:
		return (y2-y1) / (x2-x1)


def subtract_dates(beg_date, end_date):
	return (end_date - beg_date).days


def add_dates(date, num_days):
	return date + timedelta(num_days)


def calculate_AROR(principal, sell_price, num_days):
	gains = sell_price - principal
	AROR = (pow(((principal + gains) / principal), (365 / num_days)) - 1) * 100
	return AROR


def calculate_AP(bought, sold, min_date, max_date):
	G = sold - bought
	P = bought
	n = (max_date - min_date).days

	AP = pow(((P + G) / P), (365 / n)) - 1
	AP = round(AP * 100, 2)
	return AP


def average(stock_vals):
	vals = [v[1] for v in stock_vals]
	if len(vals) == 0:
		return 0

	avg = sum(vals) / len(vals)
	return round(avg, 2)


def what_if_investment_calc(principal, ticker, AROR):
	min_date, max_date = stock_dict[ticker]['dates'][0], stock_dict[ticker]['dates'][-1]
	date_diff = (max_date - min_date)
	date_diff_yrs = (date_diff.days + date_diff.seconds / 86400) / 365.2425
	gains = calculate_gains(principal, AROR, date_diff_yrs)
	print(f'${principal} used to buy {ticker} stock on {min_date} would be worth ${gains} on {max_date}, a {date_diff_yrs} year timespan')


def get_min_max(dates, start, end, arr):
	start_index = lookup_stock_val(dates, start, arr, get_index=True)
	end_index = lookup_stock_val(dates, end, arr, get_index=True)
	min_val = math.inf
	max_val = -math.inf

	for val in arr[start_index:end_index+1]:
		if val < min_val:
			min_val = val
		if val > max_val:
			max_val = val

	return min_val, max_val


def set_y_lim(ax, lookup_str, start, end):
	global stock_ax, MACD_ax, RSI_ax

	min_price, max_price = get_min_max(stock_dict[current_ticker]['dates'], start, end, stock_dict[current_ticker][lookup_str])
	ax.set_ylim(min_price - ((max_price - min_price) * 0.075), max_price + ((max_price - min_price) * 0.075))

# uses O(n) to determine correct position of top/bottom val
# could be O(log(n)) with binary search but with only showing top 10 it doesnt seem worth it
def get_top_x(pair, x):
	vals = sorted(pair, key=lambda tup: tup[1], reverse=True)
	return vals[:x]


def get_bottom_x(pair, x):
	vals = sorted(pair, key=lambda tup: tup[1])
	return vals[:x]


def calculate_gains(principal, annual_gain, num_years):
	if annual_gain == 0:
		return principal

	# initial interest gained = 0, meaning profit = principal
	comp_interest = principal

	for year in range(1, math.floor(num_years) + 1):
		comp_interest = (1 + (annual_gain / 100)) * comp_interest

	if math.floor(num_years) != num_years:
		dec_num = num_years - math.floor(num_years)
		comp_interest = (1 + ((annual_gain / 100) * dec_num)) * comp_interest

	return round(comp_interest, 2)


# uses binary search to lookup stock val
def lookup_stock_val(dates, target, arr, get_index=False):

	start = 0
	end = len(dates)-1
	while start <= end:
		mid = start + ((end - start) // 2)
		# print(type(target), type(dates[mid-1]), type(dates[mid]))

		if dates[mid] == target:
			return mid if get_index else arr[mid]
		# check if target is a weekend and get prev close price
		elif (mid-1 >= 0 and target > dates[mid-1] and target < dates[mid]):
			return mid-1 if get_index else arr[mid-1]
		elif mid+1 < len(dates) and target > dates[mid] and target < dates[mid+1]:
			return mid if get_index else arr[mid]
		elif dates[mid] > target:
			end = mid-1
		else:
			start = mid+1


def get_indicator(date, indicator):
	return lookup_stock_val(stock_dict[current_ticker]['dates'], date, stock_dict[current_ticker][indicator])


def annotate_chart(ax, ax_name, date, value):
	global annotate_boxes

	if ax_name in annotate_boxes and annotate_boxes[ax_name] is not None:
		try:
			annotate_boxes[ax_name].remove()
		except:
			pass

	arrowprops = dict(arrowstyle="wedge,tail_width=0.5", alpha=0.7, color='b')
	bbox = dict(boxstyle="round", alpha=0.7, color='b')

	y_min, y_max = ax.get_ylim()
	annotate_boxes[ax_name] = ax.annotate(
		value,
		xy=(date, (0.075 * (y_max - y_min)) + value),
		xytext=(0, 0),
		textcoords='offset points',
		size=13,
		ha='center', va="center",
		bbox=bbox,
		arrowprops=arrowprops
	)


def draw_buy_signal(ax, date, y_val):
	arrowprops = dict(arrowstyle="simple", alpha=0.7, color='g')
	bbox = dict(boxstyle="round", alpha=0.7, color='g')

	x_val = stock_dict[current_ticker]['date_to_index'][date]
	y_min, y_max = ax.get_ylim()
	ax.annotate(
		'buy',
		xy=(x_val, y_val),
		xytext=(0, -25), #(1 * (y_max - y_min))),
		textcoords='offset points',
		size=12,
		ha='center', va="center",
		bbox=bbox,
		arrowprops=arrowprops
	)


def draw_sell_signal(ax, date, y_val):
	arrowprops = dict(arrowstyle="simple", alpha=0.7, color='r')
	bbox = dict(boxstyle="round", alpha=0.7, color='r')

	x_val = stock_dict[current_ticker]['date_to_index'][date]
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


def annotate_trade_signals(ax, signals):

	for signal in signals:
		date, price, action = signal
		if ax is MACD_ax:
			y_val = lookup_stock_val(stock_dict[current_ticker]['dates'], date, stock_dict[current_ticker]['MACD'])
		elif ax is RSI_ax:
			y_val = lookup_stock_val(stock_dict[current_ticker]['dates'], date, stock_dict[current_ticker]['RSI'])
		else:
			y_val = price

		# print(action, date, y_val)
		if action == 'buy':
			draw_buy_signal(ax, date, y_val)
		else:
			assert(action == 'sell')
			draw_sell_signal(ax, date, y_val)


def draw_hover_line(ax, ax_name, x_val):
	global hover_lines

	if ax_name in hover_lines and hover_lines[ax_name] in ax.lines:
		l = ax.lines.remove(hover_lines[ax_name])
		del l

	hover_lines[ax_name] = ax.axvline(x = x_val, color='#5353ff')


def hover(event):
	global hover_line, last_date_hovered

	if event.xdata is None:
		return

	index = min(round(event.xdata), len(stock_dict[current_ticker]['index'])-1)
	date = (stock_dict[current_ticker]['dates'][index])
	# print(date)

	if date is not None:
		last_date_hovered = date

		keys = {'prices': None, 'MACD': None, 'signal': None, 'RSI': None, 'EMA 9': None, 'EMA 50': None, 'EMA 180': None}
		for key in keys:
			val = lookup_stock_val(stock_dict[current_ticker]['dates'], date, stock_dict[current_ticker][key])
			# print(val)
			if val is not None:
				val = round(val, 2)
				keys[key] = val

		if keys['prices'] is not None:
			if stock_ax:
				annotate_chart(stock_ax, 'stock_ax', round(event.xdata), keys['prices'])
				draw_hover_line(stock_ax, 'stock_ax', event.xdata)
			if MACD_ax:
				annotate_chart(MACD_ax, 'MACD_ax', round(event.xdata), keys['MACD'])
				draw_hover_line(MACD_ax, 'MACD_ax', event.xdata)
			if RSI_ax:
				annotate_chart(RSI_ax, 'RSI_ax', round(event.xdata), keys['RSI'])
				draw_hover_line(RSI_ax, 'RSI_ax', event.xdata)

	# plt.draw()


def click(event):
	global start, start_index

	if event.xdata is None:
		return
	start_index = round(event.xdata)
	start = dt.num2date(stock_dict[current_ticker]['index'][round(event.xdata)]).date()


def click_release(event):
	global start, end, start_index

	# if user released click off the graph (left or right)
	if event.xdata is None:
		if last_date_hovered is not None:
			# use last hovered date to determine if hovered off left or right of graph
			if last_date_hovered < start:
				end = min_date
			else:
				end = max_date
		else:
			end = max_date
		end_index = stock_dict[current_ticker]['date_to_index'][end]
	else:
		# index = round(event.xdata)
		# end = stock_dict[current_ticker]['dates'][index]
		end_index = round(event.xdata)
		end = dt.num2date(stock_dict[current_ticker]['index'][end_index]).date()

	# reverse start/end dates if user dragged from right to left
	if start > end:
		start, end = end, start
		start_index, end_index = end_index, start_index

	if stock_ax:
		stock_ax.set_xlim(start_index, end_index)
		set_y_lim(stock_ax, 'prices', start, end)

	if MACD_ax:
		MACD_ax.set_xlim(start_index, end_index)
		set_y_lim(MACD_ax, 'MACD', start, end)

	if RSI_ax:
		RSI_ax.set_xlim(start_index, end_index)
		set_y_lim(RSI_ax, 'RSI', start, end)

	start = end = start_index = end_index = None



def convert_todate(xval):
	if xval is None:
		return None

	date = num2date(xval).date()
	return date


def get_lookup_str(ax):
	rst = ""
	if ax is stock_ax:
		rst = 'prices'
	elif ax is MACD_ax:
		rst = 'MACD'
	elif ax is RSI_ax:
		rst = 'RSI'

	return rst



def is_inrange(date):
	min_date = stock_dict[current_ticker]['dates'][0]
	max_date = stock_dict[current_ticker]['dates'][-1]
	return date >= min_date and date <= max_date


def shift_left(ax, event):
	global start, end

	if ax:
		sub_days = 10
		x_min, x_max = map(convert_todate, ax.get_xlim())
		x_min -= timedelta(sub_days)
		x_max -= timedelta(sub_days)

		if is_inrange(x_min) and is_inrange(x_max) and x_min < x_max:
			lookup_str = get_lookup_str(ax)
			ax.set_xlim(x_min, x_max)
			set_y_lim(ax, lookup_str, x_min, x_max)

def shift_right(ax, event):
	global start, end

	if ax:
		sub_days = 10
		x_min, x_max = map(convert_todate, ax.get_xlim())
		x_min += timedelta(sub_days)
		x_max += timedelta(sub_days)

		if is_inrange(x_min) and is_inrange(x_max) and x_min < x_max:
			lookup_str = get_lookup_str(ax)
			ax.set_xlim(x_min, x_max)
			set_y_lim(ax, lookup_str, x_min, x_max)


def keypress(event):
	global next_plot

	key_pressed = event.key.lower()
	if key_pressed == 'r':
		reset(stock_ax, event)
		reset(MACD_ax, event)
		reset(RSI_ax, event)

	elif key_pressed == 'left':
		shift_left(stock_ax, event)
		shift_left(MACD_ax, event)
		shift_left(RSI_ax, event)

	elif key_pressed == 'right':
		shift_right(stock_ax, event)
		shift_right(MACD_ax, event)
		shift_right(RSI_ax, event)

	elif key_pressed == 'n':
		plt.close()
		next_plot = True


def reset(ax, event):
	start_i = stock_dict[current_ticker]['date_to_index'][min_date]
	end_i = stock_dict[current_ticker]['date_to_index'][max_date]
	if ax:
		ax.set_xlim(start_i, end_i)
		lookup_str = get_lookup_str(ax)
		set_y_lim(ax, lookup_str, min_date, max_date)


def format_date(x, pos=None):
	thisind = np.clip(int(x + 0.5), 0, N - 1)
	date = stock_dict[current_ticker]['dates'][thisind].strftime('%Y-%m-%d')
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


def stock_plot(ticker, dates, prices, MACD, signal, histogram, RSI, EMA_9, EMA_50, EMA_200, MACD_signals, EMA_signals, RSI_signals, days=0, months=0, years=0, year_num=0):
	global stock_fig, stock_ax, MACD_ax, RSI_ax, N

	# filter stock data by date range
	if year_num != 0:
		dates, stock_data = filter_year(dates, prices, MACD, signal, EMA_50, year=year_num)
	else:
		dates, stock_data = filter(dates, prices, MACD, signal, EMA_50, days=days, months=months, years=years)
	prices, MACD, signal, EMA_50 = stock_data

	# styling stuff
	plt.style.use("seaborn-dark")
	for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
		plt.rcParams[param] = '#212946'  # bluish dark grey
	for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
		plt.rcParams[param] = '0.9'  # very light grey

	plt.ion()

	stock_fig = plt.figure(figsize=(12,8))

	stock_fig.canvas.mpl_connect('motion_notify_event', hover)
	stock_fig.canvas.mpl_connect('button_press_event', click)
	stock_fig.canvas.mpl_connect('button_release_event', click_release)
	stock_fig.canvas.mpl_connect('key_press_event', keypress)

	stock_rows, MACD_rows, RSI_rows = get_plot_heights(plot_stock, plot_MACD, plot_RSI)

	N = len(prices)
	ind = [i for i in range(N)]  # np.arange(N)  # the evenly spaced plot indices

	if plot_stock:
		stock_ax = plt.subplot2grid((20,1), (0,0), rowspan=stock_rows) # rows/columns/fig_number
		stock_ax.set_title(ticker)

		# stock_ax.plot(ind, prices)
		_open, high, low, close = stock_dict[current_ticker]['open'], stock_dict[current_ticker]['high'], stock_dict[current_ticker]['low'], stock_dict[current_ticker]['close']
		data = {'Date': dates, 'Open': _open, 'High': high, 'Low': low, 'Close': close}
		df = pd.DataFrame(data, columns=['Date','Open','High','Low','Close'], index=dates)

		ap0 = [mpf.make_addplot(EMA_9, color='#F88017', ax=stock_ax),  # uses panel 0 by default
			   mpf.make_addplot(EMA_50, color='#0041C2', ax=stock_ax),  # uses panel 0 by default
			   mpf.make_addplot(EMA_200, color='#3CB043', ax=stock_ax)
				]

		mc = mpf.make_marketcolors(up='#0DB586', down='r', inherit=True)
		s = mpf.make_mpf_style(marketcolors=mc)

		mpf.plot(df, type='candle', style=s, ax=stock_ax, addplot=ap0)
		# stock_ax.plot(ind, EMA_9, '#F88017')
		# stock_ax.plot(ind, EMA_50, '#0041C2')
		# stock_ax.plot(ind, EMA_200, '#3CB043')

		stock_ax.grid(color='#2A3459')  # bluish dark grey, but slightly lighter than background
		stock_ax.xaxis.set_major_formatter(format_date)

	if plot_MACD:
		MACD_ax = plt.subplot2grid((20,1), (stock_rows, 0), rowspan=MACD_rows)

		MACD_ax.plot(ind, MACD, '#fa9933')
		MACD_ax.plot(ind, signal, '#5985ff')

		green_histogram = [val if val > 0 else 0 for val in histogram]
		red_histogram = [val if val < 0 else 0 for val in histogram]

		MACD_ax.bar(ind, green_histogram, color='green')
		MACD_ax.bar(ind, red_histogram, color='red')

		MACD_ax.grid(color='#2A3459')  # bluish dark grey, but slightly lighter than background
		MACD_ax.xaxis.set_major_formatter(format_date)


	if plot_RSI:
		RSI_ax = plt.subplot2grid((20,1), (stock_rows+MACD_rows, 0), rowspan=RSI_rows)
		RSI_ax.plot(ind, RSI, '#5985ff')
		RSI_ax.grid(color='#2A3459')  # bluish dark grey, but slightly lighter than background

	# 	# plot horizontal 30, 70 line thresholds
		RSI_ax.axhline(y=30, color='#C11B17')
		RSI_ax.axhline(y=70, color='#C11B17')

		RSI_ax.xaxis.set_major_formatter(format_date)

	stock_fig.autofmt_xdate()

	# annotate buy/sell signals
	print(f'EMA: {len(EMA_signals)}, MACD: {len(MACD_signals)}, RSI: {len(RSI_signals)}')
	if stock_ax:
		annotate_trade_signals(stock_ax, EMA_signals)
	if MACD_ax:
		annotate_trade_signals(MACD_ax, MACD_signals)
	if RSI_ax:
		annotate_trade_signals(RSI_ax, RSI_signals)


	if stock_ax:
		# stock_ax.set_xlabel('timestamp')
		stock_ax.set_ylabel('Stock Price')
		# stock_ax.set_xticks([i for i in range(len(dates))])
		# stock_ax.set_xticklabels(dates)
		stock_ax.set_xticks([])

	if MACD_ax:
		MACD_ax.set_ylabel('MACD/signal')
		MACD_ax.set_xlabel('Timestamp')
		# MACD_ax.set_xticks(dates)
		# MACD_ax.set_xticklabels(x_labels)
		# MACD_ax.set_xticks([])

	if RSI_ax:
		RSI_ax.set_ylabel('RSI Value')
		RSI_ax.set_xlabel('timestamp')
		# RSI_ax.set_xticks(filtered_dates)


	plt.show(block=True)
	# plt.draw()


def main():
	global stock_dict, current_ticker, min_date, max_date, min_index, max_index, next_plot

	stock_list = ['spy', 'tsla', 'aapl', 'msft', 'gme']
	if write_mode:
		store_stock_data(stock_list=stock_list)
	else:
		stock_dict = read_stored_stocks(stock_list=True)

		if print_to_ext_file:
			stdout_backup = sys.stdout
			log_file = open('cur_strat.log', 'w')
			sys.stdout = log_file


		MACD_ARORs, EMA_ARORs, RSI_ARORs = [], [], []
		for ticker in stock_dict:
			current_ticker = ticker
			if ticker != 'aapl':
				continue

			stock = stock_dict[ticker]
			data = stock['data']
			# data = stock_lookup(ticker)
			dates, prices, _open, high, low, close = data['dates'], data['prices'], data['open'], data['high'], data['low'], data['close']

			print(f'\n{ticker}__________________________________________')

			# dates, prices = filter_between(dates, prices, start_date=datetime(2019, 12, 1), end_date=datetime(2020, 12, 31))
			MACD, signal, histogram = get_MACD(dates, prices)
			RSI = get_RSI(dates, prices)
			# EMA_9 = get_EMA(dates, prices, 9)
			EMAs = []
			EMAs.append(get_EMA(dates, prices, 9))
			EMAs.append(get_EMA(dates, prices, 50))
			EMAs.append(get_EMA(dates, prices, 180))

			# years = 10
			start_date, end_date = datetime(2010,1,1), datetime(2012,1,1)
			dates, filtered_data = filter_between(dates, prices, _open, high, low, close, MACD, signal, histogram, RSI, EMAs[0], EMAs[1], EMAs[2], start_date=start_date, end_date=end_date)
			prices, _open, high, low, close, MACD, signal, histogram, RSI, EMAs[0], EMAs[1], EMAs[2] = filtered_data
			# if ticker didn't exist before end_date, then continue since there's no data to analyze
			if not dates:
				continue

			# print(MACD_signals)
			# print(EMA_signals)
			# write data for quick lookup when hovering over chart
			stock['dates'] = dates
			stock['prices'] = prices
			stock['open'] = _open
			stock['high'] = high
			stock['low'] = low
			stock['close'] = close

			stock['MACD'] = MACD
			stock['signal'] = signal
			stock['histogram'] = histogram
			stock['RSI'] = RSI
			stock['EMA 9'] = EMAs[0]
			stock['EMA 50'] = EMAs[1]
			stock['EMA 180'] = EMAs[2]
			stock['date_to_index'] = {d: i for i, d in enumerate(dates)}
			stock['index'] = [dt.date2num(d) for d in dates]
			stock_dict[ticker] = stock

			min_date, max_date = dates[0], dates[-1]
			min_index, max_index = stock['index'][0], stock['index'][-1]


			# it's possible that if there are no trade signals within start/end date, then this function returns None
			MACD_signals = backtest_MACD(dates, prices, MACD, signal, histogram, EMAs[-1])
			EMA_signals = backtest_EMA(dates, prices, EMAs)
			RSI_signals = backtest_RSI(dates, prices, RSI)

			MACD_signals = filter_trade_signals(MACD_signals, start_date=start_date, end_date=end_date)
			EMA_signals = filter_trade_signals(EMA_signals, start_date=start_date, end_date=end_date)
			RSI_signals = filter_trade_signals(RSI_signals, start_date=start_date, end_date=end_date)


			with open('output.txt', 'w') as f:
				prev_trade = None
				ROIs = []
				for trade in MACD_signals:
					date, price, action = trade
					if action == 'buy':
						prev_trade = trade

					if action == 'sell' and prev_trade:
						ROI = (price - prev_trade[1]) / prev_trade[1]
						ROIs.append(ROI)
						f.write(f'bought on {prev_trade[0]}, sold on {date}, profit {round(price - prev_trade[1], 2)}, ROI: {round(ROI,2)}%\n')
						prev_trade = None

				num_trades = len(MACD_signals) // 2
				avg_ROI = sum(ROIs) / len(ROIs) if len(ROIs) != 0 else 0
				gain_per_dollar = pow((1 + avg_ROI), num_trades)
				P = 1
				G = gain_per_dollar - P
				n = (max_date - min_date).days
				annualize_ROI = pow(((P + G) / P), (365 / n)) - 1
				annualize_ROI = round(annualize_ROI * 100, 2)
				f.write(f'avg ROI: {round(avg_ROI,2)}, annualized ROI: {annualize_ROI}')

			buy_hold_strat = calculate_AP(bought=prices[0], sold=prices[-1], min_date=dates[0], max_date=dates[-1])

			# get results (success rate, etc.)
			MACD_total_profit, MACD_success_rate, MACD_AROI = rank_trade_signals(MACD_signals, min_date, max_date)
			EMA_total_profit, EMA_success_rate, EMA_AROI = rank_trade_signals(EMA_signals, min_date, max_date)
			RSI_total_profit, RSI_success_rate, RSI_AROI = rank_trade_signals(RSI_signals, min_date, max_date)

			# keep track of rate of returns for ranking - top/bottom 10
			if MACD_AROI is not None:
				MACD_ARORs.append((ticker, MACD_AROI, buy_hold_strat))
			if EMA_AROI is not None:
				EMA_ARORs.append((ticker, EMA_AROI, buy_hold_strat))
			if RSI_AROI is not None:
				RSI_ARORs.append((ticker, RSI_AROI, buy_hold_strat))

			# print(prices[-1], prices[0])
			print(f'annualized rate of return for buy and hold strategy = {buy_hold_strat}%\n')
			print_trade_results(MACD_total_profit, MACD_success_rate, MACD_AROI, len(MACD_signals)/2, 'MACD')
			print_trade_results(EMA_total_profit, EMA_success_rate, EMA_AROI, len(EMA_signals)/2, 'EMA')
			print_trade_results(RSI_total_profit, RSI_success_rate, RSI_AROI, len(RSI_signals)/2, 'RSI')

			if plot_data:
				stock_plot(ticker, dates, prices, MACD, signal, histogram, RSI, EMAs[0], EMAs[1], EMAs[2], MACD_signals, EMA_signals, RSI_signals)
				while not next_plot:
					pass

			# input('press enter to go to next stock')

		# exit(0)

		avg_MACD = average(MACD_ARORs)
		avg_EMA = average(EMA_ARORs)
		avg_RSI = average(RSI_ARORs)

		best_MACD = get_top_x(MACD_ARORs, 10)
		worst_MACD = get_bottom_x(MACD_ARORs, 10)
		best_EMA = get_top_x(EMA_ARORs, 10)
		worst_EMA = get_bottom_x(EMA_ARORs, 10)
		best_RSI = get_top_x(RSI_ARORs, 10)
		worst_RSI = get_bottom_x(RSI_ARORs, 10)

		print_top_x(best_MACD, 'MACD')
		print_top_x(best_EMA, 'EMA')
		print_top_x(best_RSI, 'RSI')

		print_bottom_x(worst_MACD, 'MACD')
		print_bottom_x(worst_EMA, 'EMA')
		print_bottom_x(worst_RSI, 'RSI')

		# if len(best_AROR) != 0 and len(worst_AROR) != 0:
		# 	what_if_investment_calc(100, best_AROR[0][0], best_AROR[0][1])

		print(f'average ARORs:\nMACD = {avg_MACD}, EMA = {avg_EMA}, RSI = {avg_RSI}')

		if print_to_ext_file:
			sys.stdout = stdout_backup
			log_file.close()


if __name__ == '__main__':
	main()
