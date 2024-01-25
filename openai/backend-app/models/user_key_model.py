import sqlite3
import time
from typing import Optional

from .base_model import BaseModel

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
                password_hash VARCHAR(255),
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
    
    def create_user(self, user_id: int, username: str, password_hash: str):
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO user_key (user_id, username, password_hash, update_time)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, username, password_hash, int(time.time())),
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
    
    def validate_user(self, username: str, password_hash: str):
        res = self.fetchone(
            """
            SELECT password_hash FROM user_key WHERE username = ?
            """,
            (username,),
        )
        return (res is not None) and (res[0] == password_hash)

    def get_user_id_by_username_password_hash(self, username: str, password_hash: str) -> Optional[int]:
        res = self.fetchone(
            """
            SELECT user_id FROM user_key WHERE username = ? AND password_hash = ?
            """,
            (username, password_hash),
        )
        return res[0] if res else None

if __name__ == "__main__":
    # unittest
    conn = sqlite3.connect("unittest.db")
    user_key_model = UserKeyModel(conn)

    user_key_model.drop_tables()
    user_key_model.create_tables()

    user_id = 1
    username = "test1"
    password_hash = "test1"
    user_key_model.create_user(user_id, username, password_hash)
    res = user_key_model.get_user_id_by_username_password_hash(username, password_hash)
    print("user_id:", res)

    res = user_key_model.validate_username(username)
    print("validate_username:", res)

    res = user_key_model.validate_user(username, password_hash)
    print("validate_user:", res)

    res = user_key_model.validate_username("test2")
    print("validate_username:", res)

    res = user_key_model.validate_user(username, "test2")
    print("validate_user:", res)
    