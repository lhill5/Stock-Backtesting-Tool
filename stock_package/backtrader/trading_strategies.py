import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
import math
import financial_calcs as fin
import datetime


def buysell_minmax(stock):
    dates = stock.dates
    prices = stock.prices['high']

    df = pd.DataFrame(prices, columns=['data'])
    n = 1  # number of points to be checked before and after

    # Find local peaks
    df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                        order=n)[0]]['data']
    df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                        order=n)[0]]['data']
    df = df.rename(columns={'min': 'buy', 'max': 'sell'})

    df_trades = df.copy()
    df_trades['data'] = np.nan
    df_trades['buy'] = np.nan
    df_trades['sell'] = np.nan

    # create offset df where buy/sell indicators are +1 index from df (since minmax requires knowing future data)
    for i, row in df.iterrows():
        if i != 0 and not math.isnan(df['buy'][i-1]):
            df_trades['data'][i] = df_trades['buy'][i] = prices[i]
        if i != 0 and not math.isnan(df['sell'][i - 1]):
            df_trades['data'][i] = df_trades['sell'][i] = prices[i]

    return df, df_trades


# buy when MACD rises above signal and sell when MACD falls below signal
def buysell_MACD1(dates, prices, tech_indicators):
    MACD = tech_indicators['MACD']
    signal = tech_indicators['signal']
    histogram = tech_indicators['histogram']

    buysell_df = pd.DataFrame(columns=['buy', 'sell'], index=list(range(0, len(dates))))

    i = 0
    buy_list = []
    sell_list = []
    bought = False
    prev_MACD, prev_signal = MACD[0], signal[0]
    differences = []

    for date, price, MACD_val, signal_val, hist_val in zip(dates, prices, MACD, signal, histogram):
        MACD_difference = (MACD_val - signal_val)
        differences.append(MACD_difference)

        # if prices[i] == 64.67:
        #     breakpoint()

        buy_price = np.nan
        sell_price = np.nan

        # buy signal
        if MACD[i] > signal[i] and MACD_difference > 0.2 and not bought:
            buy_price = price
            bought = True
        # sell signal
        elif MACD[i] < signal[i] and MACD_difference < -0.2 and bought:
            sell_price = price
            bought = False

        buy_list.append(buy_price)
        sell_list.append(sell_price)

        # if action is not None:
        #     MACD_signals.append(action)
        i += 1

    # success_rate = 0 if total_trades == 0 else (successful_trades / total_trades) * 100
    # plt.plot(dates, differences)
    # plt.show()

    buysell_df['buy'] = buy_list
    buysell_df['sell'] = sell_list
    return buysell_df


# buy when histogram crosses 0 threshold, RSI less than 70 | sell when histogram below 0 and RSI > 30
def buysell_MACD2(dates, prices, tech_indicators):
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


def buysell_MACD3(dates, prices, tech_indicators):
    MACD = tech_indicators['MACD']
    signal = tech_indicators['signal']
    histogram = tech_indicators['histogram']

    buysell_df = pd.DataFrame(columns=['buy', 'sell'], index=list(range(0, len(dates))))

    i = 0
    buy_list = []
    sell_list = []
    bought = False
    prev_MACD, prev_signal = MACD[0], signal[0]
    for date, price, MACD_val, signal_val, hist_val in zip(dates, prices, MACD, signal, histogram):
        action = None
        rate_of_change = (MACD_val - prev_MACD)

        buy_price = np.nan
        sell_price = np.nan

        # buy signal
        if MACD[i] > signal[i] and not bought:
            buy_price = price
            bought = True
        # sell signal
        elif MACD[i] < signal[i] and bought:
            sell_price = price
            bought = False

        buy_list.append(buy_price)
        sell_list.append(sell_price)

        # if action is not None:
        #     MACD_signals.append(action)
        i += 1

    # success_rate = 0 if total_trades == 0 else (successful_trades / total_trades) * 100
    buysell_df['buy'] = buy_list
    buysell_df['sell'] = sell_list
    return buysell_df


def plot_results(df, df_trades):

    # plt.scatter(df.index, df['buy'], c='g')
    # plt.scatter(df.index, df['sell'], c='r')
    plt.scatter(df_trades.index, df_trades['buy'], c='g')
    plt.scatter(df_trades.index, df_trades['sell'], c='r')
    plt.plot(df.index, df['data'])
    plt.show()


def evaluate_strategy(stock, indices):
    start_date = end_date = None
    try:
        start_date, end_date = stock.dates[0], stock.dates[-1]
    except:
        breakpoint()

    num_days = (end_date - start_date).days
    # print(num_days)

    prices = stock.prices['high']

    shares = 0
    total_transactions = 0
    total_profit = 0
    annual_return = 1
    buy_queue = []
    total_return = 0
    total_bought_price = total_sold_price = 0
    first_buy_price = last_sell_price = -1

    principal = 1000
    current_balance = principal

    for i, row in indices.iterrows():
        # print(i)
        if len(row) == 3:
            data, buy_price, sell_price = row
        else:
            buy_price, sell_price = row

        # buy signal
        if not math.isnan(buy_price):
            if i < len(prices):
                if first_buy_price == -1:
                    first_buy_price = buy_price

                if len(buy_queue) == 0:
                     buy_queue.append(buy_price)

                # buy_queue = [buy_price]

                # buy_price = prices[i]
                total_bought_price += buy_price
                shares += 1

        # sell signal
        elif not math.isnan(sell_price):
            if shares == 0:
                continue
            else:
                # sell all shares
                for p in buy_queue:
                    if i < len(prices):
                        sell_price = prices[i]
                    elif i == len(prices):
                        sell_price = prices[i-1]
                    else:
                        break

                    last_sell_price = sell_price
                    total_sold_price += sell_price

                    profit = (sell_price - p) * (current_balance / p)
                    current_balance += profit
                    total_profit += profit

                    profit_ratio = (sell_price - p) / p
                    total_return += profit_ratio

                    annual_return *= (1 + profit_ratio)

                    shares -= 1
                    total_transactions += 1

                buy_queue = []
                # assert(shares == 0)

    avg_return_percent = ((annual_return ** (365. / num_days)) - 1) * 100
    CAGR = fin.calculate_AROR(principal, current_balance, num_days)

    # print(f'prices: {prices[0], prices[-1]}')
    # print(f'annual return: {annual_return}')
    # print(f'avg return percent: {avg_return_percent}')
    # print(f'ending return %: {((current_balance - principal) / principal) * 100}')
    # print(f'CAGR: {CAGR}')

    # avg_return_percent = ((cum_return ** (1. / total_transactions)) - 1) * 100
    # avg_return_gain = ((1 + (avg_return_percent / 100)) ** total_transactions) * dollar_amount_invested
    # avg_return_gain -= dollar_amount_invested  # subtract principal to get profit

    start_price = prices[0]
    end_price = prices[-1]

    buy_sell_profit = (end_price - start_price)
    buy_sell_gain = (buy_sell_profit / (start_price)) * 100
    # if stock.ticker == 'aapl':
    #     breakpoint()

    buysell_CAGR = fin.calculate_AROR(start_price=start_price, end_price=end_price, num_days=num_days)

    # total_gain = (total_profit / start_price) * 100
    # annual_gain = fin.calculate_AROR(start_price=start_price, end_price=fin.investment_return(start_price, total_gain), num_days=num_days)

    # print(f'tot transactions: {total_transactions}')

    # print(f'b&s profit per share:   {buy_sell_profit:.2f}')
    # print(f'tot profit per share:   {total_profit:.2f}')
    #
    # print(f'b&s return %: {buy_sell_gain:.2f}')
    # print(f'tot return %: {total_gain:.2f}')

    # print(f'b&s annualized return %: {buy_sell_annual_gain:.2f}')
    # print(f'tot annualized return %: {annual_gain:.2f}')

    return (total_transactions, buy_sell_profit, total_profit, buy_sell_gain, buysell_CAGR, CAGR)