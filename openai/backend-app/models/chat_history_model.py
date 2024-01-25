import json
import sqlite3
import time
from typing import Tuple

from .base_model import BaseModel

class ChatHistoryModel(BaseModel):

    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(conn)

    def create_tables(self) -> None:
        # chat history table
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                thread_id VARCHAR(255),
                conversation TEXT,
                update_time INTEGER,

                UNIQUE(thread_id)
            );
            """
        )
        self.conn.commit()
    
    def drop_tables(self) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            DROP TABLE IF EXISTS chat_history;
            """,
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def save_chat_history(self, user_id: int, thread_id: str, conversation: list) -> None:
        # insert chat history
        cursor = self.conn.cursor()
        str_conversation = json.dumps(conversation)
        now = int(time.time())
        self._execute_sql(
            cursor,
            """
            INSERT INTO chat_history (user_id, thread_id, conversation, update_time)
            VALUES (?, ?, ?, ?) 
            ON CONFLICT (thread_id) DO UPDATE SET conversation = ?, update_time = ?;
            """,
            (   
                user_id,
                thread_id,
                str_conversation,
                now,
                str_conversation,
                now,
            ),
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def get_chat_history_by_thread_id(self, thread_id: str) -> Tuple:
        # get chat history
        return self.fetchone(
            """
            SELECT * FROM chat_history
            WHERE thread_id = ?;
            """,
            (thread_id,),
            on_raise=True,
        )

    def get_chat_history_by_user_id(self, user_id: int) -> list:
        return self.fetchall(
            """
            SELECT * FROM chat_history
            WHERE user_id = ?;
            """,
            (user_id,),
            on_raise=True,
        )

if __name__ == "__main__":
    conn = sqlite3.connect("unittest.db")
    chat_history_model = ChatHistoryModel(conn)

    chat_history_model.drop_tables()
    chat_history_model.create_tables()

    user_id = 1
    thread_id = "test_thread_id"
    chat_history_model.save_chat_history(user_id, thread_id, [{"text": "test"}])
    res = chat_history_model.get_chat_history_by_thread_id(thread_id)
    print("chat history:", res)

    chat_history_model.save_chat_history(user_id, thread_id, [{"text": "test"}, {"text": "test2"}])
    res = chat_history_model.get_chat_history_by_thread_id(thread_id)
    print("chat history:", res)

    res = chat_history_model.get_chat_history_by_user_id(user_id)
    print("full chat history of user:", res)

    conn.close()