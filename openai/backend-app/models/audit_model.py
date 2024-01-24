import random
import sqlite3
import string
import time

from base_model import BaseModel

class AuditModel(BaseModel):

    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(conn)

    def create_tables(self):
        # audit log table
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                billing_period VARCHAR(255),
                thread_id VARCHAR(255),
                model VARCHAR(255),
                input_token INTEGER,
                cost_per_input_token DOUBLE,
                output_token INTEGER,
                cost_per_output_token DOUBLE,
                cost DOUBLE,
                timestamp INTEGER
            );
            """,
            commit=True,
            on_raise=True,
        )

        # audit status table
        self._execute_sql(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS audit_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                billing_period VARCHAR(255),
                sum_cost DOUBLE,
                timestamp INTEGER
            );
            """,
            commit=True,
            on_raise=True,
        )
        cursor.close()

        