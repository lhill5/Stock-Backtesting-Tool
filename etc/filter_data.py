from datetime import datetime


def filter_trade_signals(trade_signals, years=0, year_num=0, start_date=datetime.now(), end_date=datetime.now()):
	if len(trade_signals) < 2:
		return []

	cur_year = datetime.now().year

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
	start, end = -1,-1
	for i, date in enumerate(dates):
		if date >= start_date and start == -1:
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
