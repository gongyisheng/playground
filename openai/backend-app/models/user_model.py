import sqlite3
from typing import Tuple

from .base_model import BaseModel

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
                monthly_budget INTEGER,
                status INTEGER,

                UNIQUE(user_id)
            );
            """
        )
    
    def drop_tables(self) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            DROP TABLE IF EXISTS user;
            """,
            commit=True,
            on_raise=True
        )
    
    def create_user(self) -> int:
        # create a user with default monthly budget
        # return user_id
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO user (monthly_budget, status)
            VALUES (?, ?);
            """,
            (self.DEFAULT_MONTHLY_BUDGET, self.USER_STATUS_ACTIVE),
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
            on_raise=True
        )
    
    def get_user_status(self, user_id: int) -> int:
        user_info = self._get_user(user_id)
        return user_info[2]
    
    def get_user_monthly_budget(self, user_id: int) -> int:
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
            on_raise=True
        )
        cursor.close()
    
    def update_user_monthly_budget(self, user_id: int, monthly_budget: int) -> None:
        # update user monthly budget
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            UPDATE user SET monthly_budget = ? WHERE user_id = ?;
            """,
            (monthly_budget, user_id),
            commit=True,
            on_raise=True
        )
        cursor.close()

if __name__ == "__main__":
    conn = sqlite3.connect("unittest.db")
    user_model = UserModel(conn)

    user_model.drop_tables()
    user_model.create_tables()

    user_id = user_model.create_user()
    print("user_id: ", user_id)

    user_info = user_model.get_user(user_id)
    print("user_info:", user_info)

    user_model.update_user_status(user_id, UserModel.USER_STATUS_INACTIVE)
    user_status = user_model.get_user_status(user_id)
    print("user_status:", user_status)

    user_model.update_user_monthly_budget(user_id, 100)
    user_monthly_budget = user_model.get_user_monthly_budget(user_id)
    print("user_monthly_budget:", user_monthly_budget)
