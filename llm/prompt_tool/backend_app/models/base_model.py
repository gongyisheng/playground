import logging
import re
import sqlite3
from typing import Tuple

FORMAT_SQL_RE = re.compile(r"\s+")


class BaseModel:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def _format_sql(self, sql: str) -> str:
        sql = sql.replace("\n", " ").replace("\t", " ").strip()
        sql = FORMAT_SQL_RE.sub(" ", sql)
        return sql

    def _execute_sql(
        self,
        cursor: sqlite3.Cursor,
        sql: str,
        params: Tuple = None,
        commit: bool = True,
        on_raise: bool = True,
    ) -> bool:
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            if commit:
                self.conn.commit()
            logging.info(
                f"execute_sql succ. sql: [{self._format_sql(sql)}], params: [{params}], commit: [{commit}]"
            )
            return True
        except Exception as e:
            logging.error(
                f"execute_sql fail. sql: [{self._format_sql(sql)}], params: [{params}], commit: [{commit}]"
            )
            if on_raise:
                raise e
            return False

    def fetchone(self, sql: str, params: Tuple = None, on_raise: bool = True) -> Tuple:
        cursor = self.conn.cursor()
        res = self._execute_sql(cursor, sql, params, commit=False, on_raise=on_raise)
        data = None
        if res:
            data = cursor.fetchone()
        cursor.close()
        return data

    def fetchall(self, sql: str, params: Tuple = None, on_raise: bool = True) -> list:
        cursor = self.conn.cursor()
        res = self._execute_sql(cursor, sql, params, commit=False, on_raise=on_raise)
        data = None
        if res:
            data = cursor.fetchall()
        cursor.close()
        return data

    def fetchmany(
        self, sql: str, params: Tuple = None, size: int = 8, on_raise: bool = True
    ) -> list:
        cursor = self.conn.cursor()
        res = self._execute_sql(cursor, sql, params, commit=False, on_raise=on_raise)
        data = None
        if res:
            data = cursor.fetchmany(size)
        cursor.close()
        return data
