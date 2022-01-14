import logging
from time import time
from os import path, listdir
from typing import List
from msql.connection import connection


class MigrationTool:

    def __init__(self, conn_str: str, migration_dir: str, schema_table: str = "msql_migration") -> None:
        self.conn_str = conn_str
        self.migration_dir = migration_dir
        self.schema_table = schema_table

    def find_migrations(self) -> List[str]:
        return sorted([x for x in listdir(self.migration_dir) if x.endswith(".sql")])

    def install(self) -> None:
        """
        Creates database table to track schema changes
        Safe to call multiple times
        """
        with connection(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.schema_table} (
                    id INTEGER,
                    migration VARCHAR(128),
                    applied_timestamp INTEGER
                );
            """)
            conn.commit()
        logging.info(f"Migration database table created as {self.schema_table}")

    def find_applied_migrations(self) -> List[str]:
        with connection(self.conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {self.schema_table};")
            return [x[1] for x in cursor.fetchall()]

    def _run_migration(self, migration: str) -> None:
        with open(path.join(self.migration_dir, migration), "r") as f:
            sql_statement = f.read()

        try:
            with connection(self.conn_str) as conn:
                cursor = conn.cursor()
                if hasattr(cursor, "executescript"):
                    # sqlite is ret..., different
                    cursor.executescript(sql_statement)  # type: ignore
                    cursor.execute(f"INSERT INTO {self.schema_table} (migration, applied_timestamp) VALUES (?, ?);", (
                        migration,
                        time(),
                    ))
                else:
                    cursor.execute(sql_statement)
                    cursor.execute(f"INSERT INTO {self.schema_table} (migration, applied_timestamp) VALUES (%s, %s);", (
                        migration,
                        time(),
                    ))
                conn.commit()
        except Exception as e:
            logging.exception(f"Failed to perform SQL transaction\n{str(e)}")
            raise

    def apply_migrations(self) -> None:
        """
        Finds unapplied migrations and applies them
        """
        applied = self.find_applied_migrations()

        for migration in self.find_migrations():
            if migration in applied:
                continue

            try:
                self._run_migration(migration)
                logging.info(f"Applied migration: {migration}")
            except Exception as e:
                logging.exception(f"Failed to apply migration: {migration}")
                logging.exception(str(e))
                raise
