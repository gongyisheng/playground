import sqlite3
import time
import sys
from typing import Optional

sys.path.append("../../")

from backend_app.utils import setup_logger

from backend_app.models.base_model import BaseModel


class InvitationCodeModel(BaseModel):
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
            CREATE TABLE IF NOT EXISTS invitation_code (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invitation_code BLOB,
                invitation_code_status INTEGER,
                update_time INTEGER,

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
            DROP TABLE IF EXISTS invitation_code;
            """,
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def insert_invitation_code(self, invitation_code: bytes) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO invitation_code (invitation_code, invitation_code_status, update_time)
            VALUES (?, ?, ?);
            """,
            (
                invitation_code,
                self.INVITATION_CODE_STATUS_ACTIVE,
                int(time.time()),
            ),
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def validate_invitation_code(self, invitation_code: bytes) -> bool:
        res = self.fetchone(
            """
            SELECT invitation_code_status FROM invitation_code WHERE invitation_code = ?
            """,
            (invitation_code,),
            on_raise=True,
        )
        return res[0] == self.INVITATION_CODE_STATUS_ACTIVE if res else False

    def claim_invitation_code(self, invitation_code: bytes) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            UPDATE invitation_code SET invitation_code_status = ?, update_time = ? WHERE invitation_code = ? AND invitation_code_status = ?;
            """,
            (
                self.INVITATION_CODE_STATUS_EXPIRED,
                int(time.time()),
                invitation_code,
                self.INVITATION_CODE_STATUS_ACTIVE,
            ),
            commit=True,
            on_raise=True,
        )
        cursor.close()


if __name__ == "__main__":
    import logging

    setup_logger()

    # unittest
    conn = sqlite3.connect("unittest.db")
    invitation_code_model = InvitationCodeModel(conn)

    invitation_code_model.drop_tables()
    invitation_code_model.create_tables()

    invitation_code_model.insert_invitation_code(b"invitation_code_1")
    invitation_code_model.insert_invitation_code(b"invitation_code_2")

    res = invitation_code_model.validate_invitation_code(b"invitation_code_1")
    logging.info("validate code result: %s", res)

    invitation_code_model.claim_invitation_code(b"invitation_code_1")
    res = invitation_code_model.validate_invitation_code(b"invitation_code_1")
    logging.info("validate code result: %s", res)
