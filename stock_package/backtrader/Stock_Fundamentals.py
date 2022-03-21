import json
import time
import requests
import numpy as np
import pandas as pd

from global_functions import convert_str_to_date
from global_decorators import *


class StockFundamentals:
    def __init__(self, database):
        self.ALPHA_VANTAGE_API_KEY = 'SSFPLDGNSX4NAWV8'
        self.POLYGON_API_KEY = 'SEp3jMzPKmbxBD_UuJw2uYxjfWgqkoaT'
        self.SQL = database


    def get_data(self, ticker, fiscal_year=None):
        db_ticker = ticker.replace('-', '').replace('.', '')

        if self.SQL.table_exists(ticker=db_ticker, fundamental_db=True):
            data = self.get_SQL_data(db_ticker, fiscal_year=fiscal_year)
            # print('querying fundamentals')
        else:
            data = self.get_API_data(db_ticker, fiscal_year=fiscal_year)
        return data


    def get_SQL_data(self, db_ticker, fiscal_year=None):
        # todo - implement fiscal year into SQL query

        fundamental_data = self.SQL.query_fundamentals(db_ticker)
        columns = ['key', 'fiscal_year', 'time_frame', 'revenue', 'COGS', 'gross_income', 'SGA', 'EBIT', 'gross_interest_expense', 'pretax_income', 'income_tax', 'net_income', 'shareholder_net_income', 'consolidated_net_income', 'operating_income', 'EPS_basic', 'EPS_diluted', 'total_current_assets', 'total_noncurrent_assets', 'fixed_assets', 'total_assets', 'total_current_liabilities', 'total_noncurrent_liabilities', 'total_liabilities', 'common_equity', 'total_shareholders_equity', 'liabilities_and_shareholder_equity', 'operating_net_cash_flow', 'investing_net_cash_flow', 'financing_net_cash_flow', 'total_net_cash_flow']
        fund_data = np.array(fundamental_data).T.tolist()

        fund_dict = {key: vals for i, (key, vals) in enumerate(zip(columns, fund_data))}
        df = pd.DataFrame.from_dict(fund_dict)
        df.columns = columns

        len = df.shape[0]
        seqs = np.arange(len)
        df["seq"] = pd.Series(seqs)
        return df


    def get_API_data(self, db_ticker, fiscal_year=None):
        return None
        # todo - get cik, and return request API data rather than none, + error checking
        # error, data = self._request_API_data(db_ticker, cik=cik, fiscal_year=fiscal_year)
        # error, data = None, None
        # if (error):
        #     return None


    def _request_API_data(self, ticker, cik=None, fiscal_year=None):
        # todo - convert return value to pandas df

        # url = f'https://api.polygon.io/v2/reference/financials/{ticker}?limit=1&type=Y&apiKey={api_key}'
        # get_latest_url = f'https://api.polygon.io/vX/reference/financials?ticker={ticker.upper()}&include_sources=true&apiKey={api_key}'
        # url = f'https://api.polygon.io/vX/reference/financials?ticker={ticker}&timeframe=annual&order=desc&sort=filing_date&apiKey={api_key}'
        if cik:
            if fiscal_year:
                date_filter_annual_url = f'https://api.polygon.io/vX/reference/financials?cik={cik}&filing_date.gte={fiscal_year}-01-01&timeframe=annual&order=asc&limit=1&sort=filing_date&apiKey={self.POLYGON_API_KEY}'
            else:
                date_filter_annual_url = f'https://api.polygon.io/vX/reference/financials?cik={cik}&filing_date.gte=2000-01-01&timeframe=annual&order=asc&limit=100&sort=filing_date&apiKey={self.POLYGON_API_KEY}'
        else:
            if fiscal_year:
                date_filter_annual_url = f'https://api.polygon.io/vX/reference/financials?ticker={ticker}&filing_date.gte={fiscal_year}-01-01&timeframe=annual&order=asc&limit=1&sort=filing_date&apiKey={self.POLYGON_API_KEY}'
            else:
                date_filter_annual_url = f'https://api.polygon.io/vX/reference/financials?ticker={ticker}&filing_date.gte=2000-01-01&timeframe=annual&order=asc&limit=100&sort=filing_date&apiKey={self.POLYGON_API_KEY}'

        r = requests.get(date_filter_annual_url).json()

        error = None
        if r['status'] == 'ERROR':
            error = r['error']
            print(error)
            return (None, None)

        results = None
        try:
            results = r['results']
        except:
            breakpoint()

        fundamental_data = {}
        for filing in results:
            start_date = filing['start_date']
            end_date = filing['end_date']
            fiscal_year = convert_str_to_date(start_date).year

            financials = filing['financials']
            balance_sheet = financials['balance_sheet']
            cash_flow_statement = financials['cash_flow_statement']
            comprehensive_income = financials['comprehensive_income']
            income_statement = financials['income_statement']

            fiscal_year_data = {'income_statement': income_statement, 'balance_sheet': balance_sheet, 'cash_flow_statement': cash_flow_statement}
            fundamental_data[fiscal_year] = fiscal_year_data

        return error, fundamental_data


    def get_stock_list(self):
        url = f'https://api.polygon.io/v3/reference/tickers?type=CS&sort=ticker&order=asc&limit=1000&apiKey={self.POLYGON_API_KEY}'
        url = f'https://api.polygon.io/v3/reference/tickers?ticker=JPM&sort=ticker&order=asc&limit=1000&apiKey={self.POLYGON_API_KEY}'

        next_url = url

        stock_list = []
        start_time = time.time()
        counter = 0

        while next_url:
            r = requests.get(next_url)
            data = r.json()
            pretty_data = json.dumps(data, indent=4)
            print(pretty_data)
            counter += 1

            # prevent more than 5 API requests per minute
            if counter % 5 == 0:
                while time.time() - start_time < 60:
                    pass
                start_time = time.time()

            elif data['status'] == 'OK':
                # add stocks to stocks list
                stock_list.extend(data['results'])

                # if more data, get next url and continue adding stocks
                next_url = None
                if 'next_url' in data:
                    next_url = data['next_url']
                    next_url += f'&type=CS&active=true&sort=ticker&order=asc&limit=1000&apiKey={self.POLYGON_API_KEY}'

            elif data['status'] == 'ERROR':
                print('error processing API request (too many requests)')
                time.sleep(5)

        return stock_list


    # todo - implement this for API call func (backup datasource to alpha vantage)
    # def parse_POLYGON_fundamental_stock_data(self, ticker, cik=None, test_data=False):
    #     time_frame = 'annual'
    #     transformed_data = []
    #
    #     if test_data:
    #         for fiscal_year in range(2010, 2021):
    #             key = f'{fiscal_year}-{time_frame}'
    #             random_data = [key, fiscal_year, time_frame]
    #             random_numbers = [random.randint(10**6, 99**7) for i in range(28)]
    #             random_numbers[12] = random.randint(1,10)
    #             random_numbers[13] = random.randint(1, 10)
    #
    #             random_data.extend(random_numbers)
    #             transformed_data.append(tuple(random_data))
    #         return transformed_data
    #
    #     # if not test_data, get data from API call
    #     error, fundamental_data = self.fundamental_data.get_API_data(ticker, cik=cik)
    #     if fundamental_data is None:
    #         return
    #
    #     # transform data to list of tuples (for inserting into SQL table)
    #     for fiscal_year, data in fundamental_data.items():
    #         income_statement = data['income_statement']
    #         revenue = check_data(income_statement, 'revenues')
    #         COGS = check_data(income_statement, 'cost_of_revenue')
    #         gross_income = check_data(income_statement, 'gross_profit')
    #         SGA = check_data(income_statement, 'operating_expenses')
    #         EBIT = check_data(income_statement, 'operating_income_loss')
    #         gross_interest_expense = check_data(income_statement, 'interest_expense_operating')
    #         pretax_income = check_data(income_statement, 'income_loss_from_continuing_operations_before_tax')
    #         income_tax = check_data(income_statement, 'income_tax_expense_benefit')
    #         net_income = check_data(income_statement, 'net_income_loss')
    #         shareholder_net_income = check_data(income_statement, 'net_income_loss_available_to_common_stockholders_basic')
    #         consolidated_net_income = check_data(income_statement, 'net_income_loss_attributable_to_parent')
    #         operating_income = check_data(income_statement, 'operating_income_loss')
    #         EPS_basic = check_data(income_statement, 'basic_earnings_per_share', dec_places=2)
    #         EPS_diluted = check_data(income_statement, 'diluted_earnings_per_share', dec_places=2)
    #
    #         balance_sheet = data['balance_sheet']
    #         total_current_assets = check_data(balance_sheet, 'current_assets')
    #         total_noncurrent_assets = check_data(balance_sheet, 'noncurrent_assets')
    #         fixed_assets = check_data(balance_sheet, 'fixed_assets')
    #         total_assets = check_data(balance_sheet, 'assets')
    #         total_current_liabilities = check_data(balance_sheet, 'current_liabilities')
    #         total_noncurrent_liabilities = check_data(balance_sheet, 'noncurrent_liabilities')
    #         total_liabilities = check_data(balance_sheet, 'liabilities')
    #         common_equity = check_data(balance_sheet, 'equity_attributable_to_parent')
    #         total_shareholders_equity = check_data(balance_sheet, 'equity')
    #         total_equity = check_data(balance_sheet, 'equity')
    #         liabilities_and_shareholder_equity = check_data(balance_sheet, 'liabilities_and_equity')
    #
    #         cashflow = data['cash_flow_statement']
    #         investing_net_cash_flow = check_data(cashflow, 'net_cash_flow_from_investing_activities')
    #         financing_net_cash_flow = check_data(cashflow, 'net_cash_flow_from_financing_activities')
    #         operating_net_cash_flow = check_data(cashflow, 'net_cash_flow_from_operating_activities')
    #         total_net_cash_flow = check_data(cashflow, 'net_cash_flow')
    #
    #         time_frame = 'annual'
    #         key = f'{fiscal_year}-{time_frame}'
    #         transformed_data.append((key, fiscal_year, time_frame, revenue, COGS, gross_income, SGA, EBIT, gross_interest_expense, pretax_income, income_tax, net_income, shareholder_net_income, consolidated_net_income, operating_income, EPS_basic, EPS_diluted, total_current_assets, total_noncurrent_assets, fixed_assets, total_assets, total_current_liabilities, total_noncurrent_liabilities, total_liabilities, common_equity, total_shareholders_equity, liabilities_and_shareholder_equity, operating_net_cash_flow, investing_net_cash_flow, financing_net_cash_flow, total_net_cash_flow))
    #
    #     return transformed_data


    def print(self, data, all=False):
        df = data

        if not all:
            print(df)
        else:
            # prints all rows/cols
            with pd.option_context('display.max_rows', None,
                                   'display.max_columns', None,
                                   'display.precision', 3):
                print(df)

