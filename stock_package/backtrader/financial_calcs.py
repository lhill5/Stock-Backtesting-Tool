import math
import datetime


def rank_trade_signals(trade_signals, min_date, max_date):
	if trade_signals is None or len(trade_signals) < 2:
		return [], None, None

	total_profit = 0
	successful_trades, total_trades = 0, 0
	bought, sold = 0, 0
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
	if x2 - x1 == 0:
		return 100
	else:
		return (y2 - y1) / (x2 - x1)


def subtract_dates(beg_date, end_date):
	return (end_date - beg_date).days


def add_dates(date, num_days):
	return date + datetime.timedelta(num_days)


def calculate_AROR(start_price, end_price, num_days):
	gains = end_price - start_price
	AROR = (pow(((start_price + gains) / start_price), (365 / num_days)) - 1) * 100
	return AROR


def investment_return(principal, percentage):
	rst = principal * (1 + (percentage / 100))
	return rst

def calculate_AP(bought, sold, min_date, max_date):
	G = sold - bought
	P = bought
	n = (max_date - min_date).days

	AP = pow(((P + G) / P), (365 / n)) - 1
	AP = round(AP * 100, 2)
	return AP


def average(values):
	if len(values) == 0:
		return 0

	avg = sum(values) / len(values)
	return round(avg, 2)


# def what_if_investment_calc(principal, ticker, AROR):
# 	min_date, max_date = stock_dict[ticker]['dates'][0], stock_dict[ticker]['dates'][-1]
# 	date_diff = (max_date - min_date)
# 	date_diff_yrs = (date_diff.days + date_diff.seconds / 86400) / 365.2425
# 	gains = calculate_gains(principal, AROR, date_diff_yrs)
# 	print(f'${principal} used to buy {ticker} stock on {min_date} would be worth ${gains} on {max_date}, a {date_diff_yrs} year timespan')


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


# finds the start date that is 'x' trading days ago from 'date'
def trading_days_ago(date, num_days):
	# figure out difference of current weekday from monday (friday - monday = 4 trading days)
	days_since_monday = date.weekday() # figures of day # of week (monday=0, sunday=6)
	if num_days < days_since_monday:
		return date - datetime.timedelta(days=num_days)

	num_days -= days_since_monday
	day_count = days_since_monday

	# then take num_days / 5 == # weeks to subtract
	num_trading_weeks = num_days // 5
	num_days -= num_trading_weeks * 5
	day_count += num_trading_weeks * 7
	assert(num_days < 5)

	# move to friday so that there's 4 days to subtract after num_days // 5 == 0
	if num_days > 0:
		day_count += 3  # sat. and sun. and 1 day for moving back 1 trading day
		num_days -= 1

	day_count += num_days

	start_date = date - datetime.timedelta(days=day_count)
	return start_date


# rounds date to previous weekday (if weekend then find closes weekday)
def round_date(date):
	day_num = date.weekday()
	# if weekend
	if day_num in [5,6]:
		return date - datetime.timedelta(days=day_num-4) # friday = 4
	return date


def get_last_trading_day():
	today = datetime.date.today()
	weekday = today.weekday()
	# if today is a weekend
	if weekday > 4:
		rst = today - datetime.timedelta(days=weekday-4)
	else:
		rst = today

	# if market hasn't closed yet then latest trading day is yesterday
	ts = datetime.datetime.now()
	hour, minute = ts.hour, ts.minute
	# if it's M-F and market hasn't closed yet get last trading date
	if hour < 16 and weekday <= 4:
		# if it's monday, then last trading day was friday
		if weekday == 0:
			rst = rst - datetime.timedelta(days=3)
		else:
			rst = rst - datetime.timedelta(days=1)

	return rst


def convert_str_to_date(str):
	format = "%Y-%m-%d"
	dt_object = datetime.datetime.strptime(str, format).date()
	return dt_object


def divide(a, b):
	if b != 0:
		return a / b
	else:
		return 0

