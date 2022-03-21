import pandas_market_calendars as mcal
import datetime


class Calendar:
    def __init__(self):
        self.holiday_lookup = self.get_holiday_lookup()
        self.date_lookup = self.get_date_lookup()
        self.index_lookup = self.get_index_lookup()
        # print('test')


    def get_date_lookup(self):
        i = -1

        date_to_index = {}
        for d in self.daterange(datetime.date(1960,1,1), datetime.date(2050,1,1)):
            if d.weekday() < 5 and d not in self.holiday_lookup:
                i += 1
            if i != -1:
                date_to_index[d] = i
        return date_to_index


    # returns dictionary where key=index, value=date
    def get_index_lookup(self):
        i = -1

        index_to_date = {}
        for d in self.daterange(datetime.date(1960,1,1), datetime.date(2050,1,1)):
            if d.weekday() < 5 and d not in self.holiday_lookup:
                i += 1
                index_to_date[i] = d
        return index_to_date


    def get_holiday_lookup(self):
        nyse = mcal.get_calendar('NYSE')
        holidays = nyse.holidays().holidays
        holiday_dict = {}
        for holiday_date in holidays:
            # convert from numpy date to datetime.date
            holiday_date = holiday_date.item()
            holiday_dict[holiday_date] = 1
        return holiday_dict


    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days) + 1):
            yield start_date + datetime.timedelta(n)


if __name__ == '__main__':
    calendar = Calendar()

