import sqlite3
import sys
import time
from typing import Optional

sys.path.append("../../")

from backend_app.utils import setup_logger

from backend_app.models.base_model import BaseModel


class UserKeyModel(BaseModel):
    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(conn)

    def create_tables(self):
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS user_key (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username VARCHAR(255),
                password BLOB,
                update_time INTEGER,

                UNIQUE(user_id)
                UNIQUE(username)
            );
            """,
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def drop_tables(self) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            DROP TABLE IF EXISTS user_key;
            """,
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def create_user(self, user_id: int, username: str, password: bytes) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO user_key (user_id, username, password, update_time)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, username, password, int(time.time())),
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def validate_username(self, username: str) -> bool:
        res = self.fetchone(
            """
            SELECT username FROM user_key WHERE username = ?
            """,
            (username,),
        )
        return res is not None

    def validate_user(self, username: str, password: bytes) -> bool:
        res = self.fetchone(
            """
            SELECT password FROM user_key WHERE username = ?
            """,
            (username,),
        )
        return (res is not None) and (res[0] == password)

    def get_user_id_by_username_password(
        self, username: str, password: bytes
    ) -> Optional[int]:
        res = self.fetchone(
            """
            SELECT user_id FROM user_key WHERE username = ? AND password = ?
            """,
            (username, password),
        )
        return res[0] if res[0] else None

    def update_password_by_username(self, username: str, password: bytes) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            UPDATE user_key SET password = ?, update_time = ?
            WHERE username = ?
            """,
            (password, int(time.time()), username),
            commit=True,
            on_raise=True,
        )
        cursor.close()


if __name__ == "__main__":
    import logging

    setup_logger()

    # unittest
    conn = sqlite3.connect("unittest.db")
    user_key_model = UserKeyModel(conn)

    user_key_model.drop_tables()
    user_key_model.create_tables()

    user_id = 1
    username = "test1"
    password = b"test1"
    user_key_model.create_user(user_id, username, password)
    res = user_key_model.get_user_id_by_username_password(username, password)
    logging.info("user_id: %s", res)

    res = user_key_model.validate_username(username)
    logging.info("validate_username: %s", res)

    res = user_key_model.validate_user(username, password)
    logging.info("validate_user: %s", res)

    res = user_key_model.validate_username("test2")
    logging.info("validate_username: %s", res)

    res = user_key_model.validate_user(username, b"test2")
    logging.info("validate_user: %s", res)

    user_key_model.update_password_by_username(username, b"test3")
    res = user_key_model.validate_user(username, b"test3")
    logging.info("validate_user: %s", res)
