import random
import string
import sqlite3
import sys
import time
from typing import Optional

sys.path.append("../../")

from backend_app.utils import setup_logger

from backend_app.models.base_model import BaseModel


class SessionModel(BaseModel):
    # session status
    SESSION_STATUS_ACTIVE = 0
    SESSION_STATUS_EXPIRED = 1

    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(conn)

    def create_tables(self) -> None:
        # session table
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS session (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                session_id VARCHAR(255),
                status INTEGER,
                last_active_time INTEGER,

                UNIQUE(user_id),
                UNIQUE(session_id)
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
            DROP TABLE IF EXISTS session;
            """,
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def generate_session_id(self) -> str:
        # random 64 bit string
        random.seed(int(time.time()))
        return "".join(
            random.choice(string.ascii_letters + string.digits) for _ in range(64)
        )

    def create_session(self, user_id: int) -> str:
        session_id = self.generate_session_id()
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO session (user_id, session_id, status, last_active_time)
            VALUES (?, ?, ?, ?) 
            ON CONFLICT(user_id) DO UPDATE SET session_id = ?, status = ?, last_active_time = ?;
            """,
            (
                user_id,
                session_id,
                self.SESSION_STATUS_ACTIVE,
                int(time.time()),
                session_id,
                self.SESSION_STATUS_ACTIVE,
                int(time.time()),
            ),
            commit=True,
            on_raise=True,
        )
        cursor.close()
        return session_id

    def validate_session(self, session_id: str) -> bool:
        res = self.fetchone(
            """
            SELECT status, last_active_time FROM session WHERE session_id = ?
            """,
            (session_id,),
            on_raise=True,
        )
        if res is None:
            return False
        else:
            status = res[0]
            return status == self.SESSION_STATUS_ACTIVE

    def expire_session(self, session_id: str) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            UPDATE session SET status = ? WHERE session_id = ?
            """,
            (self.SESSION_STATUS_EXPIRED, session_id),
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def get_user_id_by_session(self, session_id: str) -> Optional[int]:
        res = self.fetchone(
            """
            SELECT user_id FROM session WHERE session_id = ?
            """,
            (session_id,),
            on_raise=True,
        )
        return res[0] if res[0] else None


if __name__ == "__main__":
    import logging

    setup_logger()

    conn = sqlite3.connect("unittest.db")
    session_model = SessionModel(conn)

    session_model.drop_tables()
    session_model.create_tables()

    user_id = 1
    session_id = session_model.create_session(user_id)
    logging.info(session_id)
    res = session_model.validate_session(session_id)
    logging.info("validate session: %s", res)

    session_model.expire_session(session_id)
    res = session_model.validate_session(session_id)
    logging.info("validate session: %s", res)

    _user_id = session_model.get_user_id_by_session(session_id)
    logging.info("user_id: %s", _user_id)

    conn.close()
