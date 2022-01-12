import datetime
import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i+2], 16) for i in (0, 2, 4))


def lighter(color, percent):
    if '#' in color:
        color = hex_to_rgb(color)

    '''assumes color is rgb between (0, 0, 0) and (255, 255, 255)'''
    color = np.array(color)
    white = np.array([255, 255, 255])
    vector = white-color
    return color + vector * percent


def convert_epoch_to_date(epoch):
    return datetime.datetime.fromtimestamp(epoch)


def convert_str_to_date(str):
    format = "%Y-%m-%d"
    dt_object = datetime.datetime.strptime(str, format).date()
    return dt_object


def print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
        print(df)


if __name__ == '__main__':
    hex_color = "#2f2f2f"
    rgb_color = hex_to_rgb(hex_color)
    percent_lighter = lighter(rgb_color, 0.2)
    print(percent_lighter)

