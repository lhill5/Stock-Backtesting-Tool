import pdb


def get_3_line_strike(df, bullish=False, bearish=False):
    rst = []
    if bearish:
        prev_red_close = 0
        red_candle_count = 0
        for i in range(len(df)):
            green_candle = df['green_candle'][i]
            red_candle = df['red_candle'][i]
            open_price = df['open'][i]
            close = df['close'][i]
            is_pattern = 0

            # red candle
            # pdb.set_trace()
            if red_candle:
                if prev_red_close == 0:
                    red_candle_count += 1
                elif close < prev_red_close:
                    red_candle_count += 1
                else:
                    red_candle_count = 0
                prev_red_close = close
            # green candle
            else:
                if red_candle_count == 3 and open_price <= prev_red_close and close > df.loc[i-3, 'high']:
                    is_pattern = 1
                red_candle_count = 0
                prev_red_close = 0

            rst.append(is_pattern)

    elif bullish:
        prev_green_close = 0
        green_candle_count = 0
        for i in range(len(df)):
            green_candle = df['green_candle'][i]
            red_candle = df['red_candle'][i]
            open_price = df['open'][i]
            close = df['close'][i]
            is_pattern = 0

            # red candle
            if green_candle:
                if prev_green_close == 0:
                    green_candle_count += 1
                elif close > prev_green_close:
                    green_candle_count += 1
                else:
                    green_candle_count = 0
                prev_green_close = close
            # green candle
            else:
                if green_candle_count == 3 and open_price >= prev_green_close and close < df.loc[i-3, 'low']:
                    is_pattern = 1
                green_candle_count = 0
                prev_green_close = 0

            rst.append(is_pattern)
    return rst

