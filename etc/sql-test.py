import mysql.connector
from mysql.connector import Error
import pandas as pd


def connect_to_database(host_name, user_name, user_password, db_name=None):
    connection = None
    try:
        if db_name:
            connection = mysql.connector.connect(
                host=host_name,
                user=user_name,
                passwd=user_password,
                database=db_name
            )
        else:
            connection = mysql.connector.connect(
                host=host_name,
                user=user_name,
                passwd=user_password,
            )
        print("MySQL Database connection successful")

    except Error as err:
        print(f"Error: '{err}'")

    return connection


def create_database(query, pw):
    connection = connect_to_database("localhost", "root", pw)
    cursor = connection.cursor()

    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as err:
        print(f"Error: '{err}'")


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit() # write to db
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")


def multiline_query(connection, query, column_values):
    cursor = connection.cursor()
    try:
        cursor.executemany(query, column_values)
        connection.commit()  # write to db
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")


def read_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        result = cursor.fetchall() # read-only from db
        return result
    except Error as err:
        print(f"Error: '{err}'")



def main():
    pw = 'mssqlserver44'
    db = 'test'

    # create_database(connection, "CREATE DATABASE test")
    # connection = connect_to_database("localhost", "root", pw, db)

    create_person_table = """
    CREATE TABLE person (
        person_id INT PRIMARY KEY,
        name VARCHAR(20) NOT NULL,
        age INT NOT NULL,
        gender VARCHAR(1)
    );
    """

    person_data = """
    INSERT INTO person VALUES
    (1, 'Landon', 21, 'M'),
    (2, 'Paige', 52, 'F'),
    (3, 'Logan', 24, 'M'),
    (4, 'Ron', 58, 'M');
    """

    # execute_query(connection, create_person_table)
    # execute_query(connection, person_data)

    show_data = """
    SELECT *
    FROM person
    """
    # data = read_query(connection, show_data)
    # for row in data:
    #     for col_val in row:
    #         print(col_val, end=" ")
    #     print()

    db = 'stock_data'
    ticker = 'AAPL'
    # first create the database if it doesn't already exist
    create_database(f"CREATE DATABASE {db}", pw)

    # connect to that new/existing database
    connection = connect_to_database("localhost", "root", pw, db)

    create_table_query = f"""
    CREATE TABLE {ticker} (
        date DATE PRIMARY KEY,
        open DECIMAL(7,2),
        high DECIMAL(7,2),
        low DECIMAL(7,2),
        close DECIMAL(7,2),
        adj_close DECIMAL(7,2),
        volume BIGINT
    );
    """
    # create table in database if it doesn't already exist, if it does then return
    error_code = execute_query(connection, create_table_query)
    if error_code == -1:
        return

    df = pd.read_csv(f'stock_data/{ticker}.csv')
    column_values = df.values.tolist()

    insert_data_query = f"""
    INSERT INTO {ticker}
    (date, open, high, low, close, adj_close, volume) 
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    # insert data into that table
    multiline_query(connection, insert_data_query, column_values)


if __name__ == '__main__':
    main()
