
tradeable_stocks = []
with open('tradeable_stock_list.txt') as f:
    tradeable_stocks = f.readlines()

stocks = []
with open('stock_list.txt') as f:
    stocks = f.readlines()

etfs = []
with open('etf_list.txt') as f:
    etfs = f.readlines()

print(f'stocks: {len(stocks)}')
print(f'etfs: {len(etfs)}')
print(f'tradeable stocks: {len(tradeable_stocks)}')

tradeable_stocks = [t.strip() for t in tradeable_stocks]
stocks = stocks + etfs
stocks = [s.strip() for s in stocks]

stocks = set(stocks)
tradeable_stocks = set(tradeable_stocks)

intersect = stocks & tradeable_stocks
print(f'stocks: {len(stocks)}')
print(f'tradeable stocks: {len(tradeable_stocks)}')
print(f'intersect: {len(intersect)}')

not_tradeable_stocks = stocks ^ tradeable_stocks
print(not_tradeable_stocks)

find_stock = ['RDS-A']
find_stock = set(find_stock)
result = find_stock & tradeable_stocks
print(result)

