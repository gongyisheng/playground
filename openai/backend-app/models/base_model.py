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
    
    def _execute_sql(self, cursor: sqlite3.Cursor, sql: str, params: Tuple = None, commit: bool = True, on_raise: bool = True) -> bool:
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            if commit:
                self.conn.commit()
            print("execute_sql succ. sql: [%s], params: [%s], commit: [%s]" % (self._format_sql(sql), params, commit))
            return True
        except Exception as e:
            print("execute_sql fail. sql: [%s], params: [%s], commit: [%s]" % (self._format_sql(sql), params, commit))
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
    
    def fetchmany(self, sql: str, params: Tuple = None, size: int = 8, on_raise: bool = True) -> list:
        cursor = self.conn.cursor()
        res = self._execute_sql(cursor, sql, params, commit=False, on_raise=on_raise)
        data = None
        if res:
            data = cursor.fetchmany(size)
        cursor.close()
        return data