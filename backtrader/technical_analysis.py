from financial_calcs import average
import math


def get_MACD(prices):
	fast_EMA_days = 12
	slow_EMA_days = 26
	difference = 0
	signal_days = 9

	fast_EMA_factor = 2 / (fast_EMA_days + 1)
	slow_EMA_factor = 2 / (slow_EMA_days + 1)
	signal_factor = 2 / (signal_days + 1)

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
	assert (len(fast_EMA[17:]) == len(slow_EMA))

	signal = [sum(difference[:9]) / 9]
	for i, val in enumerate(difference[9:]):
		prev_val = signal[i]
		cur_val = ((val - prev_val) * signal_factor) + prev_val
		signal.append(cur_val)

	histogram = [dif - sig for dif, sig in zip(difference[8:], signal)]

	difference = difference[8:]

	left_fill = [None for _ in range(33)]
	return difference, signal, histogram


def get_EMA(prices, num_days):

	EMA = [average(prices[:num_days])]
	prev_val = EMA[0]

	factor = 2 / (num_days + 1)

	for val in prices[num_days:]:
		cur_val = ((val - prev_val) * factor) + prev_val
		EMA.append(cur_val)
		prev_val = cur_val

	return EMA


def get_SMA(prices, num_days):
	SMA = []

	for i in range(len(prices) - num_days + 1):
		SMA.append(average(prices[i:num_days+i]))

	return SMA


def get_RSI(prices):

	RSI = []
	RSI_period = 14
	prev_close = prices[0]
	gains, losses = [], []

	for close_price in prices[1:RSI_period + 1]:
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

	RSI = [RSI_step1]  # first RSI value

	prev_avg_gain = average_gains
	prev_avg_loss = average_losses

	prev_close = prices[15]
	prev_RSI = RSI_step1
	for close in prices[16:]:
		change = close - prev_close
		if change > 0:
			gain = change
			loss = 0
		else:
			loss = abs(change)
			gain = 0

		avg_up_movement = ((prev_avg_gain * (RSI_period - 1)) + gain) / RSI_period
		avg_down_movement = ((prev_avg_loss * (RSI_period - 1)) + loss) / RSI_period
		relative_strength = avg_up_movement / avg_down_movement if avg_down_movement != 0 else prev_RSI
		RSI_step2 = 100 - (100 / (1 + relative_strength))

		prev_close = close
		prev_avg_gain = avg_up_movement
		prev_avg_loss = avg_down_movement
		prev_RSI = RSI_step2
		RSI.append(RSI_step2)

	return RSI


def backtest_MACD(dates, prices, tech_indicators):
	MACD = tech_indicators['MACD']
	signal = tech_indicators['signal']
	histogram = tech_indicators['histogram']
	RSI = tech_indicators['RSI']
	EMAs = tech_indicators['EMA']

	MACD_signals = []
	prev_MACD, prev_signal = MACD[0], signal[0]
	prev_hist = histogram[0]

	i = 1
	bought = False
	for date, price, MACD_val, signal_val, hist_val, RSI_val in zip(dates[1:], prices[1:], MACD[1:], signal[1:], histogram[1:], RSI[1:]):
		action = None
		rate_of_change = (MACD_val - prev_MACD)

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


def backtest_EMA(dates, prices, tech_indicators):
	EMA = tech_indicators['EMA']

	if len(EMA) < 2:
		return []

	EMA_signals = []
	bought = False

	for i in range(len(dates)):
		action = None

		# diffs = (EMAs[0][i] - EMAs[-1][i]) / prices[i]
		min_last_month = min(EMA[50][i-365:i]) if i-365 >= 0 else math.inf

		if None not in EMA:
			# if not bought and all(EMA1[i] > EMA2[i] for EMA1, EMA2 in zip(EMAs, EMAs[1:])):
			rate_of_change = ((EMA[9][i] - EMA[200][i]) / EMA[200][i]) * 100
			if not bought and rate_of_change >= 5 and EMA[150][i] > EMA[200][i]: #all(EMA1[i] > EMA2[i] for EMA1, EMA2 in zip(EMAs, EMAs[1:])):
				action = (dates[i], prices[i], 'buy')
				bought = True
			elif bought and EMA[150][i] < EMA[200][i]:
				# if EMAs[0][i] < min_last_month:
				action = (dates[i], prices[i], 'sell')
				bought = False

		if action is not None:
			EMA_signals.append(action)

	return EMA_signals


def backtest_RSI(dates, prices, tech_indicators):
	RSI = tech_indicators['RSI']

	RSI_signals = []
	bought = False
	for i, date in enumerate(dates):
		action = None

		if RSI[i] is None:
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


def get_tech_indicator(indicator, dates, prices):
	if indicator == 'MACD':
		return get_MACD(dates, prices)
	elif 'SMA' in indicator or 'EMA' in indicator:
		n = int(indicator.split('_')[1]) # EMA_9, EMA_50, etc. gets number of days (n)
		if 'SMA' in indicator:
			return get_EMA(dates, prices, n)
		elif 'EMA' in indicator:
			return get_SMA(dates, prices, n)
	elif indicator == 'RSI':
		return get_RSI(dates, prices)

