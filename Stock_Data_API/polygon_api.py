import requests
from stock_package.backtrader.random_functions import convert_epoch_to_date, convert_str_to_date
import json


api_key = 'SEp3jMzPKmbxBD_UuJw2uYxjfWgqkoaT'

def format_data(data):
    return round(float(data), 2)


def polygon_get_data(ticker, start_date, end_date):
    start_date = str(start_date)
    end_date = str(end_date)
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?apiKey={api_key}'
    r = requests.get(url)
    data = r.json()
    if data['resultsCount'] == 0:
        return []

    data = data['results']
    data = {str(convert_epoch_to_date(d['t']/1000).date()): [d['o'], d['h'], d['l'], d['c']] for d in data}
    for key in data.keys():
        data[key] = list(map(format_data, data[key]))
    return data


def alpha_get_data(ticker, start_date, end_date):
    api_key = 'SSFPLDGNSX4NAWV8'

    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}'
    r = requests.get(url)
    data = r.json()
    data = data['Time Series (Daily)']
    data = {date: [val for val in list(vals.values())[:4]] for date, vals in sorted(data.items()) if convert_str_to_date(date) >= start_date and convert_str_to_date(date) <= end_date}
    for key in data.keys():
        data[key] = list(map(format_data, data[key]))
    return data


def compare_data(data1, data2):
    differences = []
    for date in data1.keys():
        if date not in data2:
            differences.append(date)
        else:
            list1 = data1[date]
            list2 = data2[date]
            if len(list1) != len(list2) or len(list1) != sum([1 for l1, l2 in zip(list1, list2) if l1 == l2]):
                differences.append(date)
    return differences


def get_financials(ticker):
    # url = f'https://api.polygon.io/v2/reference/financials/{ticker}?limit=1&type=Y&apiKey={api_key}'
    # get_latest_url = f'https://api.polygon.io/vX/reference/financials?ticker={ticker.upper()}&include_sources=true&apiKey={api_key}'
    url = f'https://api.polygon.io/vX/reference/financials?ticker={ticker}&timeframe=annual&order=desc&sort=filing_date&apiKey={api_key}'
    # date_filter_annual_url = f'https://api.polygon.io/vX/reference/financials?ticker={ticker}&filing_date.lte=2016-01-01&timeframe=annual&order=asc&limit=5&sort=filing_date&apiKey={api_key}'

    r = requests.get(url).json()
    results = r['results']
    latest_result = results[0]

    start_date = latest_result['start_date']
    end_date = latest_result['end_date']
    fiscal_year = latest_result['fiscal_year']

    financials = latest_result['financials']
    balance_sheet = financials['balance_sheet']
    cash_flow_statement = financials['cash_flow_statement']
    comprehensive_income = financials['comprehensive_income']
    income_statement = financials['income_statement']

    json_print(cash_flow_statement)

    return r


def get_stock_info(ticker):
    url = f'https://api.polygon.io/vX/reference/tickers/{ticker}?apiKey={api_key}'
    r = requests.get(url)
    data = r.json()
    return data


# def get_income_statement(item):
#     annual_statement = {}
#
#     revenue = item['revenues']
#     COGS = item['costOfRevenue']
#     depreciation_and_amortization = item['depreciationAmortizationAndAccretion']
#     gross_profit = revenue - COGS
#
#     SGA_expense = item['operatingExpenses']
#     R_and_D_expense = item['researchAndDevelopmentExpense']
#     other_SGA_expense = item['sellingGeneralAndAdministrativeExpense']
#     operating_expenses = SGA_expense + R_and_D_expense + other_SGA_expense
#
#     income_from_operations = gross_profit - operating_expenses
#
#     EBIT = item['earningBeforeInterestTaxesUSD']
#     operating_income = item['grossProfit'] - item['operatingExpenses']
#
#     income_tax_expense = item['incomeTaxExpense']
#     net_income_manual = income_from_operations-income_tax_expense
#     net_income = item['netIncome']
#     print(net_income_manual, net_income)
#
#     stock_split = 4
#     eps = item['earningsPerBasicShare']/stock_split
#     diluted_eps = item['earningsPerDilutedShare']/stock_split
#     outstanding_shares = net_income / eps
#
#
# def get_balance_sheet(item):
#     # current assets
#     cash_and_equivalents = item['cashAndEquivalents']
#     current_investments = item['investmentsCurrent']
#     cash_and_current_investments = cash_and_equivalents + current_investments
#     accounts_receivable = item['tradeAndNonTradeReceivables']
#     inventory = item['inventory']
#     total_current_assets_calc = cash_and_current_investments + accounts_receivable + inventory
#     total_current_assets = item['assetsCurrent']
#     asset_turnover = item['assetTurnover']
#     print(total_current_assets_calc, total_current_assets)
#
#     # non-current assets
#     net_property_plant_equipment = item['propertyPlantEquipmentNet']
#     non_current_investments = item['investmentsNonCurrent']
#     intangibles_and_goodwill = item['goodwillAndIntangibleAssets']
#     total_non_current_assets_calc = net_property_plant_equipment + non_current_investments + intangibles_and_goodwill
#     total_non_current_assets = item['assetsNonCurrent']
#     print(total_non_current_assets_calc, total_non_current_assets)
#
#     # total-assets
#     total_assets = total_current_assets + total_non_current_assets
#     total_assets_calc = item['assets']
#     print(total_assets, total_assets_calc)
#
#     # current liabilities
#     short_term_debt = item['debtCurrent']
#     accounts_payable = item['tradeAndNonTradePayables']
#     total_current_liabilities = item['currentLiabilities']
#     other_current_liabilities = total_current_liabilities - short_term_debt - accounts_payable
#
#     current_ratio = item['currentRatio']
#
#     # non-current liabilities
#     long_term_debt = item['debtNonCurrent']
#     total_non_current_liabilities = item['liabilitiesNonCurrent']
#
#     # total liabilities
#     total_liabilities_calc = total_current_liabilities + total_non_current_liabilities
#     total_liabilities = item['totalLiabilities']
#     liabilites_asset_ratio = total_liabilities / total_assets
#     print(total_liabilities_calc, total_liabilities)
#
#     # equity
#     shareholders_equity = item['shareholdersEquity']
#     retained_earnings = item['accumulatedRetainedEarningsDeficit']
#     equity_assets_ratio = shareholders_equity / total_assets
#
#     liabilities_and_equity = total_liabilities + shareholders_equity
#
#     # ensures balance sheet balances (assets=liabilities)
#     assert(total_assets == liabilities_and_equity)


def json_print(dict):
    json_dict = json.dumps(dict, indent=4)
    print(json_dict)


def test():
    url = f'https://api.polygon.io/v2/reference/financials/AAPL?limit=5&type=YA&sort=-reportPeriod&apiKey={api_key}'
    r = requests.get(url)
    data = r.json()
    return data


if __name__ == '__main__':
    ticker = 'AAPL'
    # start_date = datetime.date(2019,8,16)
    # end_date = datetime.date(2021,8,14)

    finances = get_financials(ticker)
    if finances['status'] == 'OK':
        finances_dict = finances['results']
        for row in finances['results']:
            finances_json = json.dumps(row, indent=4)
            with open('old_polygon_finances.txt', 'a') as outfile:
                outfile.write(finances_json)


        income_statement = {}
        for item in [f for f in finances_dict if f['period'] == 'Y']:
            report_date = item['reportPeriod']
            year = report_date[:4]
            income_statement[year] = {}

            revenue = item['revenues']

            # annual_statement = get_annual_income_statement(item)

            # income_statement[year] = annual_statement
            # income_statement[year][1] = first_quarterly_report
            # income_statement[year][2] = second_quarterly_report
            # income_statement[year][3] = third_quarterly_report
            # income_statement[year][4] = fourth_quarterly_report

    # alpaca_data = alpha_get_data(ticker, start_date, end_date)
    # polygon_data = polygon_get_data(ticker, start_date, end_date)
    # differences = compare_data(alpaca_data, polygon_data)
    # print(alpaca_data)
    # print(f'differences: {differences}')
    # if len(differences) != 0:
    #     print(alpaca_data)
    #     print(polygon_data)
