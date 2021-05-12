import math

def calculate_gains(principal, annual_gain, num_years):
	if annual_gain == 0:
		return principal

	# initial interest gained = 0, meaning profit = principal
	comp_interest = principal

	for year in range(1, math.floor(num_years) + 1):
		comp_interest = (1 + (annual_gain / 100)) * comp_interest

	if math.floor(num_years) != num_years:
		dec_num = num_years - math.floor(num_years)
		print(dec_num)
		comp_interest = (1 + ((annual_gain / 100) * dec_num)) * comp_interest

	return round(comp_interest, 2)


principal = 100
percent = 11621.88
years = 23.92
rst = calculate_gains(principal, percent, years)

print(f'principal = {principal}, result = ${rst}')