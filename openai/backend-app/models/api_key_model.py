import sqlite3
import time
from typing import Optional

from base_model import BaseModel

class ApiKeyModel(BaseModel):
    
    # invitation code status
    INVITATION_CODE_STATUS_ACTIVE = 0
    INVITATION_CODE_STATUS_EXPIRED = 1
    
    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(conn)

    def create_tables(self):
        # api key database
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS api_key (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                api_key VARCHAR(255),
                invitation_code VARCHAR(255),
                invitation_code_status INTEGER,
                update_time INTEGER,

                UNIQUE(api_key)
                UNIQUE(invitation_code)
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
            DROP TABLE IF EXISTS api_key;
            """,
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def insert_api_key_invitation_code(self, api_key: str, invitation_code: str) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO api_key (api_key, invitation_code, invitation_code_status, update_time)
            VALUES (?, ?, ?, ?);
            """,
            (api_key, invitation_code, self.INVITATION_CODE_STATUS_ACTIVE, int(time.time())),
            commit=True,
            on_raise=True,
        )
        cursor.close()
    
    def validate_invitation_code(self, invitation_code: str) -> bool:
        res = self.fetchone(
            """
            SELECT invitation_code_status FROM api_key WHERE invitation_code = ?
            """,
            (invitation_code,),
            on_raise=True,
        )
        return res[0] == self.INVITATION_CODE_STATUS_ACTIVE if res else False
    
    def claim_invitation_code(self, user_id: int, invitation_code: str) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            UPDATE api_key SET user_id = ?, invitation_code_status = ?, update_time = ? WHERE invitation_code = ? AND invitation_code_status = ?;
            """,
            (user_id, self.INVITATION_CODE_STATUS_EXPIRED, int(time.time()), invitation_code, self.INVITATION_CODE_STATUS_ACTIVE),
            commit=True,
            on_raise=True,
        )
        cursor.close()
    
    def get_api_key_by_user(self, user_id) -> Optional[str]:
        res = self.fetchone(
            """
            SELECT api_key FROM api_key WHERE user_id = ? AND invitation_code_status = ?;
            """,
            (user_id, self.INVITATION_CODE_STATUS_EXPIRED),
            on_raise=True,
        )
        return res[0] if res else None

if __name__ == "__main__":
    # unittest
    conn = sqlite3.connect("unittest.db")
    api_key_model = ApiKeyModel(conn)

    api_key_model.drop_tables()
    api_key_model.create_tables()

    api_key_model.insert_api_key_invitation_code("api_key_1", "invitation_code_1")
    api_key_model.insert_api_key_invitation_code("api_key_2", "invitation_code_2")

    res = api_key_model.validate_invitation_code("invitation_code_1")
    print("validate code result:", res)

    user_id = 1
    api_key_model.claim_invitation_code(user_id, "invitation_code_1")
    res = api_key_model.validate_invitation_code("invitation_code_1")
    print("validate code result:", res)
    res = api_key_model.get_api_key_by_user(user_id)
    print("api_key:", res)
