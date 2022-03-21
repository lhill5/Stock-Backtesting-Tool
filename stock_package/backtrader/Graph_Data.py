import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource
import tech_candlestick_patterns as candlestick_pattern
from financial_calcs import divide
import trading_strategies as strat


# used for converting Stock class data into Bokeh formatted data, usable in plotting functions
# each stock will have its own GraphData class
class GraphData:
    # list of stocks graph data (CAGR for example)
    stocks_trade_results = None

    def __init__(self, stock, ticker_list, trading_strategies):
        self.stock = stock
        self.ticker = stock.ticker
        self.date_to_index = stock.date_to_index
        self.ticker_list = ticker_list
        self.trading_strategies = trading_strategies

        # additional data
        self.tech_indicators = stock.tech_indicators
        self.fundamental_data = stock.fundamentals
        self.df = self.get_stock_df()
        self.fund_df = self.get_fundamental_df()
        self.init_intrinsic_value_metrics()

        # indicates when to buy or sell depending for various strategies
        self.moving_averages = self.tech_indicators['EMA'].keys()
        self.buysell_results_dict, _ = strat.get_trading_results(self.stock, self.df['seq'], self.trading_strategies)
        # appends stock's trading strat results to class attribute
        self.add_trading_strategy_results()

        self.len = self.df.shape[0]
        self.start, self.end = 0, self.len - 1
        self.start_date, self.end_date = (self.stock.dates[0]), (self.stock.dates[-1])
        self.y_limits = self.get_starting_y_limits()

        self.ticker_list_df = self.get_ticker_list_df()
        self.stock_metrics_df = self.get_stock_metrics_df()

        # covert pandas dfs to datasource (compatible dataType for Bokeh plots)
        self.stock_source = ColumnDataSource(ColumnDataSource.from_df(self.df))
        self.fundamental_source = ColumnDataSource(ColumnDataSource.from_df(self.fund_df))
        self.buysell_results_source = ColumnDataSource(ColumnDataSource.from_df(self.stocks_trade_results))

        # self.minmax_rst_source = ColumnDataSource(ColumnDataSource.from_df(self.minmax_rst_df))
        self.ticker_list_source = ColumnDataSource(ColumnDataSource.from_df(self.ticker_list_df))
        self.stock_metrics_source = ColumnDataSource(ColumnDataSource.from_df(self.stock_metrics_df))

        # user for interactivity tools (draw support/resistance lines, draw drag-highlight feature to zoom into plot area, click button to toggle between draw lines / zoom features)
        self.draw_lines_source = ColumnDataSource(data=dict(x=[], y=[]))
        self.draw_rect_source = ColumnDataSource(data=dict(x=[], y=[], width=[], height=[]))
        self.button_source = ColumnDataSource(data=dict(value=[True]))

        # relies on setting up data above
        self.calculate_intrinsic_value()

    # coverts stock data into a pandas dataframe
    def get_stock_df(self):
        df = pd.DataFrame.from_dict(self.stock.date_ohlcv)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'adj close', 'volume']

        df_len = df.shape[0]
        seqs = np.arange(df_len)
        df["seq"] = pd.Series(seqs)
        # df["date"] = pd.to_datetime(df["date"])

        # transform date to pandas timestamp (must be this otherwise data doesn't showup on bokeh chart)
        df["date"] = pd.to_datetime(df["date"])
        df['str_date'] = [str(d.date()) for d in df['date']]

        # add new columns (technical indicators)
        for indicator, values in self.tech_indicators.items():
            if 'EMA' in indicator or 'SMA' in indicator:
                for num_days, val in values.items():
                    df[f'{indicator}_{num_days}'] = val
            else:
                df[indicator] = values
        # adds useful metrics for current stock
        add_column_metrics(df, df_len)
        return df

    def get_fundamental_df(self):
        # fundamental data
        columns = ['key', 'fiscal_year', 'time_frame', 'revenue', 'COGS', 'gross_income', 'SGA', 'EBIT', 'gross_interest_expense', 'pretax_income', 'income_tax', 'net_income', 'shareholder_net_income', 'consolidated_net_income', 'operating_income', 'EPS_basic', 'EPS_diluted', 'total_current_assets', 'total_noncurrent_assets', 'fixed_assets', 'total_assets', 'total_current_liabilities', 'total_noncurrent_liabilities', 'total_liabilities', 'common_equity', 'total_shareholders_equity', 'liabilities_and_shareholder_equity', 'operating_net_cash_flow', 'investing_net_cash_flow', 'financing_net_cash_flow', 'total_net_cash_flow', 'seq']
        fund_data = np.array(self.fundamental_data).T.tolist()
        i = 3
        for _ in fund_data[3:]:
            if 'EPS' not in columns[i]:
                fund_data[i] = [int(v / 10 ** 6) for v in fund_data[i]]
            # add commas to digits (1000 -> 1,000) for readability
            fund_data[i] = [f'{v:,}' for v in fund_data[i]]
            i += 1

        # convert to pandas df
        fund_dict = {key: vals for i, (key, vals) in enumerate(zip(columns, fund_data))}
        df = pd.DataFrame.from_dict(fund_dict)
        df.columns = columns

        len = df.shape[0]
        seqs = np.arange(len)
        df["seq"] = pd.Series(seqs)
        return df

    def add_trading_strategy_results(self):

        indicator_names = self.trading_strategies
        # -- add strat --
        stock = self.stock
        if len(stock.dates) == 0:
            self.logger.info(f'cannot calculate return results for {self.ticker} stock, no data')
            return pd.DataFrame()

        # loop creates a pandas df for each strategy, minmax, MACD, EMA, etc.
        strategies_eval = []
        for ind_name in indicator_names:
            strat_eval = pd.DataFrame(columns=['ticker', f'{ind_name} total transactions', f'{ind_name} buy/sell profit', f'{ind_name} total profit', f'{ind_name} buy/sell gain', f'{ind_name} buy/sell CAGR', f'{ind_name} CAGR'])

            # gets buy/sell/dates/seq cols where trade was executed for given strategy
            trades = self.buysell_results_dict[ind_name]
            if trades is None:
                continue

            # returns list of useful info such as total_transactions, profit, etc.
            eval_rst = list(strat.evaluate_strategy(stock, trades))
            eval_rst = [round(val, 2) for val in eval_rst]
            eval_rst.insert(0, self.ticker.upper())

            strat_eval.loc[0, :] = list(eval_rst)
            strategies_eval.append(strat_eval)

        trade_results = strategies_eval[0]
        for strat_rst_df in strategies_eval[1:]:
            trade_results = trade_results.reset_index().merge(strat_rst_df, on="ticker", how="inner").set_index('index')

        # appends individual stock's trade results to class attribute (table of trade results for each stock)
        if self.stocks_trade_results is None:
            self.stocks_trade_results = trade_results
        else:
            self.stocks_trade_results = self.stocks_trade_results.append(trade_results)

    def get_ticker_list_df(self):
        stocks_data = {'stocks': self.ticker_list}
        df = pd.DataFrame.from_dict(stocks_data)
        df['intrinsic_value'] = [0 for i in range(len(self.ticker_list))]
        return df

    def get_stock_metrics_df(self):
        stock_data = {'stock': [self.ticker.upper()]}
        df = pd.DataFrame.from_dict(stock_data)
        df['intrinsic_value'] = [self.intrinsic_value]
        df['intrinsic_value_per_share'] = [self.intrinsic_value_per_share]
        return df

    def get_starting_y_limits(self):
        start_i = self.start
        end_i = self.end

        y_min = min(self.stock.low[start_i:end_i + 1])
        y_max = max(self.stock.high[start_i:end_i + 1])

        # 5 percent offset on upper/lower limit of graph
        offset = 0.05
        return y_min - offset * y_min, y_max + offset * y_max

    def convert_data(self, data):
        pass

    def init_intrinsic_value_metrics(self):
        # intrinsic value metrics
        # _______________________
        # assumptions
        self.BYFCF = 1000
        self.GR = 6
        self.DR = 10
        self.shares_outstanding = 1000
        self.LGR = 3

        self.FCF = []
        self.DF = []

        self.DFCF = []
        self.DPCF = 0

        self.intrinsic_value = 0
        self.intrinsic_value_per_share = 0

    def calculate_intrinsic_value(self):
        # calculate free cash flow for next 10 years
        self.FCF = [self.BYFCF * pow(1 + (self.GR / 100), n) for n in range(1, 11)]
        # print(self.FCF)
        # calculate discount factor for next 10 years
        self.DF = [pow(1 + (self.DR / 100), n) for n in range(1, 11)]
        # print(self.DF)
        # calculate discounted free cash flow from above calculations
        self.DFCF = [FCF / DF for FCF, DF in zip(self.FCF, self.DF)]
        # print(self.DFCF)
        # calculate discounted perpetuity free cash flow beyond 10 years
        self.DPCF = divide((self.BYFCF * pow((1 + (self.GR / 100)), 11) * (1 + (self.LGR / 100))), ((self.DR - self.LGR) / 100)) * (divide(1, pow(1 + (self.DR / 100), 11)))
        # print(self.DPCF)
        # finally calculate the intrinsic value based on above calculations
        self.intrinsic_value = round(sum(self.DFCF) + self.DPCF, 2)
        # print(self.intrinsic_value)
        # another useful metric for showing the intrinsic value per share and comparing this with current share price ($)
        self.intrinsic_value_per_share = round(divide(self.intrinsic_value, self.shares_outstanding), 2)
        self.update_stock_metrics()

    def update_stock_metrics(self):
        self.stock_metrics_source.data['intrinsic_value'] = [self.intrinsic_value]
        self.stock_metrics_source.data['intrinsic_value_per_share'] = [self.intrinsic_value_per_share]


def add_column_metrics(df, df_size):
    # creates percent_change (close - prev_close) / prev_close column, where first % change is 0
    df['percent_change'] = [0] + [round(divide(df.loc[i, 'close'] - df.loc[i - 1, 'close'], df.loc[i - 1, 'close']) * 100, 2) for i in range(1, df_size)]
    df['percent_change_str'] = [str(change) + '%' for change in df['percent_change']]
    df['percent_change_color'] = ["#12C98C" if change >= 0 else "#F2583E" for change in df['percent_change']]

    df['green_candle'] = [1 if df.loc[i, 'close'] >= df.loc[i, 'open'] else 0 for i in range(df_size)]
    df['red_candle'] = [1 if df.loc[i, 'close'] < df.loc[i, 'open'] else 0 for i in range(df_size)]

    df['EMA9-200_percent'] = [divide((df.loc[i, 'EMA_9'] - df.loc[i, 'EMA_200']), df.loc[i, 'EMA_200']) * 100 for i in range(df_size)]
    df['candle_above_EMA9'] = [1 if low > EMA_9 else -1 for low, EMA_9 in zip(df['low'], df['EMA_9'])]

    df['bullish_3_line_strike'] = candlestick_pattern.get_3_line_strike(df, bullish=True)
    df['bearish_3_line_strike'] = candlestick_pattern.get_3_line_strike(df, bearish=True)

    # EMA indicators
    df['EMA_trend'] = [1 if i != 0 and (EMA9 - df.EMA_9[i - 1]) > 0 else 0 for i, EMA9 in enumerate(df.EMA_9)]
    # -- ema trend
    ema_selloff = [0]
    max_EMA9 = df['EMA_9'][0]
    selloff = 0
    i = 0
    for EMA_9_val, EMA_50_val, EMA_200_val in zip(df['EMA_9'][1:], df['EMA_50'][1:], df['EMA_200'][1:]):
        max_EMA9 = max(max_EMA9, EMA_9_val)
        ema_selloff.append(((max_EMA9 - df['EMA_9'][i]) / df['EMA_9'][i]) * 100)
        # if (((max_EMA9 - EMA_9[i]) / EMA_9[i]) * 100) > 10:
    df['EMA_selloff'] = ema_selloff

