import random
import string
import sqlite3
import time

from base_model import BaseModel

class SessionModel(BaseModel):

    # session status
    SESSION_STATUS_ACTIVE = 0
    SESSION_STATUS_EXPIRED = 1

    # session settings
    SESSION_EXPIRE_TIME = 60 * 60 * 24 * 30 # 30 days

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

                UNIQUE(user_id, session_id)
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
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(64))

    def create_session(self, user_id: int) -> str:
        session_id = self.generate_session_id()
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO session (user_id, session_id, status, last_active_time)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, session_id, self.SESSION_STATUS_ACTIVE, int(time.time())),
            commit=True,
            on_raise=True,
        )
        cursor.close()
        return session_id

    def validate_session(self, user_id: int, session_id: str) -> bool:
        res = self.fetchone(
            """
            SELECT status, last_active_time FROM session WHERE user_id = ? AND session_id = ?
            """,
            (user_id, session_id),
            on_raise=True,
        )
        if res is None:
            return False
        else:
            status = res[0]
            last_active_time = res[1]
            if status == self.SESSION_STATUS_ACTIVE:
                if int(time.time()) - last_active_time <= self.SESSION_EXPIRE_TIME:                    
                    return True
                else:
                    self.expire_session(user_id, session_id)
                    return False
            else:
                return False
    
    def expire_session(self, user_id: int, session_id: str) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            UPDATE session SET status = ? WHERE user_id = ? AND session_id = ?
            """,
            (self.SESSION_STATUS_EXPIRED, user_id, session_id),
            commit=True,
            on_raise=True,
        )
        cursor.close()

if __name__ == '__main__':
    conn = sqlite3.connect('unittest.db')
    session_model = SessionModel(conn)

    session_model.drop_tables()
    session_model.create_tables()

    user_id = 1
    session_id = session_model.create_session(user_id)
    print(session_id)
    res = session_model.validate_session(user_id, session_id)
    print("validate session: ", res)

    session_model.expire_session(user_id, session_id)
    res = session_model.validate_session(user_id, session_id)
    print("validate session: ", res)

    conn.close()


        