import sqlite3
import sys
import time
from typing import List, Tuple

sys.path.append("../../")

from backend_app.utils import setup_logger

from backend_app.models.base_model import BaseModel

class FileModel(BaseModel):
    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(conn)

    def create_tables(self) -> None:
        # file table
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS file (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                file_name VARCHAR(255),
                file_size_mb DOUBLE,
                insert_time INTEGER,
                last_use_time INTEGER,

                UNIQUE(user_id, file_name)
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
            DROP TABLE IF EXISTS file;
            """,
            commit=True,
            on_raise=True,
        )
        cursor.close()
    
    def create_file(self, user_id: int, file_name: str, file_size_mb: float)-> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO file (user_id, file_name, file_size_mb, insert_time, last_use_time)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, file_name) DO UPDATE SET insert_time=?, last_use_time = ?;
            """,
            (user_id, file_name, file_size_mb, int(time.time()), int(time.time()),
             int(time.time()), int(time.time())),
            commit=True,
            on_raise=True,
        )
        cursor.close()
    
    def get_file_names_by_user_id(self, user_id: int, limit: int = 10) -> List[Tuple[str]]:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            SELECT file_name
            FROM file
            WHERE user_id = ?
            ORDER BY last_use_time DESC;
            """,
            (user_id,),
        )
        result = cursor.fetchall()
        cursor.close()
        return result
    
    def get_file_name_by_user_id_file_name(self, user_id: int, file_name: str):
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            SELECT file_name
            FROM file
            WHERE user_id = ? AND file_name = ?;
            """,
            (user_id, file_name),
        )
        result = cursor.fetchone()
        cursor.close()
        return result
    
    def update_last_use_time(self, user_id: int, file_name: str) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            UPDATE file SET last_use_time = ? WHERE user_id = ? AND file_name = ?;
            """,
            (int(time.time()), user_id, file_name),
            commit=True,
            on_raise=True,
        )
        cursor.close()
    
    def delete_file_by_user_id(self, user_id: int) -> bool:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            DELETE FROM file WHERE user_id = ?;
            """,
            (user_id,),
            commit=True,
            on_raise=True,
        )
        cursor.close()
        return True
    
    def delete_file_by_user_id_file_name(self, user_id: int, file_name: str) -> bool:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            DELETE FROM file WHERE user_id = ? AND file_name = ?;
            """,
            (user_id, file_name),
            commit=True,
            on_raise=True,
        )
        cursor.close()
        return True

if __name__ == "__main__":
    import logging

    setup_logger()

    # unittest
    conn = sqlite3.connect("unittest.db")
    file_model = FileModel(conn)

    file_model.drop_tables()
    file_model.create_tables()

    file_model.create_file(1, "test_file1", "0.995")
    time.sleep(1)
    file_model.create_file(1, "test_file2", "0.995")
    time.sleep(1)
    file_model.create_file(1, "test_file3", "0.995")

    res = file_model.get_file_names_by_user_id(1)
    print(res)

    res = file_model.get_file_name_by_user_id_file_name(1, "test_file1")
    print(res)

    file_model.update_last_use_time(1, "test_file1")
    res = file_model.get_file_names_by_user_id(1)
    print(res)

    file_model.delete_file_by_user_id_file_name(1, "test_file1")
    res = file_model.get_file_names_by_user_id(1)
    print(res)

    file_model.delete_file_by_user_id(1)
    res = file_model.get_file_names_by_user_id(1)
    print(res)
