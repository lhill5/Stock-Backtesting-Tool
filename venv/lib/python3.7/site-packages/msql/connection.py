from __future__ import annotations
from typing import cast, Any
from typing_extensions import Protocol
from msql.cursor import Cursor

import sqlite3


class Connection(Protocol):

    def cursor(self) -> Cursor:
        ...

    def close(self) -> None:
        ...

    def commit(self) -> None:
        ...

    def __enter__(self) -> Connection:
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...


# sadly we need to hold in memory connections
global_sqlite_memory_conn = None


def conn_sqlite(conn_str: str) -> Connection:
    global global_sqlite_memory_conn

    # this transforms "sqlite://:memory:" => ":memory:"
    name = conn_str[len('sqlite://'):]

    conn = cast(Connection, sqlite3.connect(name))
    conn.row_factory = sqlite3.Row  # type: ignore

    # if memory, we need to hold one connection
    if name == ":memory:":
        if global_sqlite_memory_conn is None:
            global_sqlite_memory_conn = conn
        return global_sqlite_memory_conn
    else:
        return conn


factories = {"sqlite": conn_sqlite}


try:
    import psycopg2
    import psycopg2.extras

    def conn_postgres(conn_str: str) -> Connection:
        return cast(Connection, psycopg2.connect(conn_str, cursor_factory=psycopg2.extras.DictCursor))

    factories["postgresql"] = conn_postgres
except ModuleNotFoundError:
    pass


def conn_unknown(conn_str: str) -> Connection:
    raise RuntimeError("Unsupported DB type in connection string")


def connection(conn_str: str) -> Connection:
    """
    Main factory that creates connections.
    Depending on connection string it will use library to create actual connection.

    :raises RuntimeError if unsupported DB type is used in connection string
    """
    db_type = conn_str.split(':')[0]

    return factories.get(db_type, conn_unknown)(conn_str)
