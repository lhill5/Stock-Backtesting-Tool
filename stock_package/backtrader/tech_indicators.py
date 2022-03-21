from financial_calcs import average
import pandas as pd
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


def get_ADX(high_prices, low_prices, close_prices):
	pos_DMs = []
	neg_DMs = []
	TRs = []

	i = 1
	for cur_high, cur_low, cur_close in zip(high_prices[1:15], low_prices[1:15], close_prices[1:15]):
		prev_high = high_prices[i-1]
		prev_low = low_prices[i-1]
		prev_close = close_prices[i-1]

		pos_DM = cur_high - prev_high
		neg_DM = prev_low - cur_low

		# fix if negative/positive
		pos_DM = max(pos_DM, 0) if pos_DM > neg_DM else 0
		neg_DM = max(neg_DM, 0) if neg_DM > pos_DM else 0

		TR1 = max(cur_high - cur_low, abs(cur_high - prev_close), abs(cur_low - prev_close))

		TRs.append(TR1)
		pos_DMs.append(pos_DM)
		neg_DMs.append(neg_DM)
		i += 1

	first_TR14 = sum(TRs)
	first_pos_DM14 = sum(pos_DMs)
	first_neg_DM14 = sum(neg_DMs)

	prior_TR14 = first_TR14
	prior_pos_DM14 = first_pos_DM14
	prior_neg_DM14 = first_neg_DM14

	smoothed_TR14s = [first_TR14]
	smoothed_pos_DM14s = [first_pos_DM14]
	smoothed_neg_DM14s = [first_neg_DM14]

	for cur_high, cur_low, cur_close in zip(high_prices[15:], low_prices[15:], close_prices[15:]):
		prev_high = high_prices[i - 1]
		prev_low = low_prices[i - 1]
		prev_close = close_prices[i - 1]

		current_pos_DM1 = cur_high - prev_high
		current_neg_DM1 = prev_low - cur_low

		current_pos_DM1 = max(current_pos_DM1, 0) if current_pos_DM1 > current_neg_DM1 else 0
		current_neg_DM1 = max(current_neg_DM1, 0) if current_neg_DM1 > current_pos_DM1 else 0

		pos_DMs.append(current_pos_DM1)
		neg_DMs.append(current_neg_DM1)

		current_TR1 = max(cur_high - cur_low, abs(cur_high - prev_close), abs(cur_low - prev_close))
		TRs.append(current_TR1)
		# DM = pos_DM if pos_DM > neg_DM else neg_DM

		smoothed_TR14 = prior_TR14 - (prior_TR14/14) + current_TR1
		smoothed_pos_DM14 = prior_pos_DM14 - (prior_pos_DM14/14) + current_pos_DM1
		smoothed_neg_DM14 = prior_neg_DM14 - (prior_neg_DM14/14) + current_neg_DM1

		smoothed_TR14s.append(smoothed_TR14)
		smoothed_pos_DM14s.append(smoothed_pos_DM14)
		smoothed_neg_DM14s.append(smoothed_neg_DM14)

		prior_TR14 = smoothed_TR14
		prior_pos_DM14 = smoothed_pos_DM14
		prior_neg_DM14 = smoothed_neg_DM14

		i += 1

	pos_DI14s = [100 * (pos_DM14 / TR14) for pos_DM14, TR14 in zip(smoothed_pos_DM14s, smoothed_TR14s)]
	neg_DI14s = [100 * (neg_DM14 / TR14) for neg_DM14, TR14 in zip(smoothed_neg_DM14s, smoothed_TR14s)]
	DXs = [100 * (abs(pos_DI14 - neg_DI14) / (pos_DI14 + neg_DI14)) for pos_DI14, neg_DI14 in zip(pos_DI14s, neg_DI14s)]

	first_ADX = average(DXs[:14])
	prior_ADX = first_ADX

	ADXs = [first_ADX]
	for DX in DXs[14:]:
		ADX = ((prior_ADX * 13) + DX) / 14
		ADXs.append(ADX)
		prior_ADX = ADX

	return neg_DI14s[13:], pos_DI14s[13:], ADXs


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


if __name__ == '__main__':

	df = pd.read_excel('cs-adx.xls', header=1)
	df = df.loc[:, ~df.columns.isin(['Unnamed: 0', 'Unnamed: 1'])]
	high = df['High'].to_list()
	low = df['Low'].to_list()
	close = df['Close'].to_list()
	ADX = df['ADX'].dropna().to_list()

	get_ADX(high, low, close)
