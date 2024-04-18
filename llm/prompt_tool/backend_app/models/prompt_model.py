import sqlite3
import sys

sys.path.append("../../")

from backend_app.utils import setup_logger

from backend_app.models.base_model import BaseModel


class PromptModel(BaseModel):
    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(conn)

    def create_tables(self) -> None:
        # saved prompt table
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS saved_prompt (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                prompt_name VARCHAR(255),
                prompt_content TEXT,
                prompt_note TEXT,

                UNIQUE(user_id, prompt_name)
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
            DROP TABLE IF EXISTS saved_prompt;
            """,
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def save_prompt(
        self, user_id, prompt_name: str, prompt_content: str, prompt_note: str
    ) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO saved_prompt (user_id, prompt_name, prompt_content, prompt_note)
            VALUES (?, ?, ?, ?) 
            ON CONFLICT (user_id, prompt_name) DO UPDATE SET prompt_content = ?, prompt_note = ?;
            """,
            (
                user_id,
                prompt_name,
                prompt_content,
                prompt_note,
                prompt_content,
                prompt_note,
            ),
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def get_prompts_by_user_id(self, user_id: int) -> list:
        return self.fetchall(
            """
            SELECT prompt_name, prompt_content, prompt_note FROM saved_prompt WHERE user_id = ?;
            """,
            (user_id,),
            on_raise=True,
        )


if __name__ == "__main__":
    import logging

    setup_logger()

    conn = sqlite3.connect("unittest.db")
    prompt_model = PromptModel(conn)

    prompt_model.drop_tables()
    prompt_model.create_tables()

    user_id = 1
    prompt_model.save_prompt(user_id, "test1", "test1", "test1")
    res = prompt_model.get_prompts_by_user_id(user_id)
    logging.info("user prompts:", res)

    prompt_model.save_prompt(user_id, "test2", "test2", "test2")
    res = prompt_model.get_prompts_by_user_id(user_id)
    logging.info("user prompts:", res)

    prompt_model.save_prompt(user_id, "test2", "test2-new", "test2-new")
    res = prompt_model.get_prompts_by_user_id(user_id)
    logging.info("user prompts:", res)

    conn.close()
