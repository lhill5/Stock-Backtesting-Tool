from SQL_DB import SQL_DB

database = SQL_DB(None, update=False)


delete_stocks = [stock for stock in database.stocks if '&' in stock]
for delete_stock in delete_stocks:
    table_name = f'{delete_stock}_table'
    database.delete_table()