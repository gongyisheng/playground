import sqlite3
import sys
from typing import Tuple

sys.path.append("../../")

from backend_app.utils import setup_logger

from backend_app.models.base_model import BaseModel


class UserModel(BaseModel):
    # user status
    USER_STATUS_ACTIVE = 0
    USER_STATUS_INACTIVE = 1

    # budget
    DEFAULT_MONTHLY_BUDGET = 5

    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(conn)

    def create_tables(self) -> None:
        # user tabble
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS user (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                status INTEGER,

                UNIQUE(user_id)
            );
            """,
        )

    def drop_tables(self) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            DROP TABLE IF EXISTS user;
            """,
            commit=True,
            on_raise=True,
        )

    def create_user(self) -> int:
        # create a user with default monthly budget
        # return user_id
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO user (status)
            VALUES (?);
            """,
            (self.USER_STATUS_ACTIVE,),
            commit=True,
            on_raise=True,
        )
        user_id = cursor.lastrowid
        cursor.close()
        return user_id

    def _get_user(self, user_id: int) -> Tuple:
        # return user info
        return self.fetchone(
            """
            SELECT * FROM user WHERE user_id = ?;
            """,
            (user_id,),
            on_raise=True,
        )

    def get_user_status(self, user_id: int) -> int:
        user_info = self._get_user(user_id)
        return user_info[1]

    def update_user_status(self, user_id: int, status: int) -> None:
        # update user status
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            UPDATE user SET status = ? WHERE user_id = ?;
            """,
            (status, user_id),
            commit=True,
            on_raise=True,
        )
        cursor.close()


if __name__ == "__main__":
    import logging

    setup_logger()

    conn = sqlite3.connect("unittest.db")
    user_model = UserModel(conn)

    user_model.drop_tables()
    user_model.create_tables()

    user_id = user_model.create_user()
    logging.info("user_id: %s", user_id)

    user_info = user_model._get_user(user_id)
    logging.info("user_info: %s", user_info)

    user_model.update_user_status(user_id, UserModel.USER_STATUS_INACTIVE)
    user_status = user_model.get_user_status(user_id)
    logging.info("user_status: %s", user_status)
