import pandas as pd
from os import environ
from sqlalchemy import create_engine
import sqlalchemy as db
from sqlalchemy.orm import scoped_session, sessionmaker
# import pyodbc

from SQL_DB import SQL_DB
from global_decorators import *
from StockList import StockList
from event_logger import Logger


def main():
    host_name = "localhost"
    user_name = "root"
    user_password = 'mssqlserver44'
    stock_db = 'stock_data'
    port = '3306'

    SQL = SQL_DB()
    logger = Logger()
    stock_list = StockList(SQL, logger)

    URI = f'mysql+pymysql://{user_name}:{user_password}@{host_name}:{port}/{stock_db}'
    # URI = f'mssql+pyodbc://{user_name}:{user_password}@{host_name}:{port}/{stock_db}'

    engine = create_engine(URI)
    connection = engine.connect()
    ticker = 'MMM'

    tickers = stock_list.tickers
    start_time = time.time()
    total_time = 0
    for ticker in tickers:
        inner_start_time = time.time()
        sql_connector_test(ticker, SQL)
        inner_end_time = time.time()
        # print(f'{inner_end_time - inner_start_time} seconds')
        total_time += (inner_end_time - inner_start_time)
        break

    end_time = time.time()
    # print(f'{len(tickers)} stocks took -> {end_time - start_time} seconds')
    # print(f'total time: {total_time}')
    # sql_alchemy_test(engine, connection, ticker)


def sql_connector_test(ticker, SQL):
    data = SQL.query_prices(ticker, start_date='2011-01-01', end_date='2022-03-07')


@timer
def sql_alchemy_test(engine, connection, ticker):
    # metadata = db.MetaData()
    # query = db.select([db.Table('AAPL_table', metadata, autoload=True, autoload_with=engine)])
    # result_proxy = connection.execute(query)
    # result_set = result_proxy.fetchall()
    #
    # flag = True
    # data = []
    # while flag:
    #     partial_results = result_proxy.fetchmany(50)
    #     if (partial_results == []):
    #         flag = False
    #     data.append(partial_results)
    #
    # result_proxy.close()
    table_df = pd.read_sql(
        f"SELECT * FROM {ticker}_table where date > '2011-01-01'",
        con=engine,
        index_col='date'
    )

@timer
def sql_pyodbc_test(conn, ticker):
    pass
    # cnxn = pyodbc.connect('DRIVER={Devart ODBC Driver for MySQL};User ID=myuserid;Password=mypassword;Server=myserver;Database=mydatabase;Port=myport;String Types=Unicode')


if __name__ == '__main__':
    main()
