import dbm
import sqlite3
import time

from base_model import BaseModel
from pricing_model import PricingModel


class AuditModel(BaseModel):

    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(conn)
        self.pricing_model = PricingModel(conn)
        self.status_cache = dbm.open("audit-status", "c")

    def create_tables(self):
        # audit log table
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                billing_period VARCHAR(255),
                thread_id VARCHAR(255),
                model VARCHAR(255),
                input_token INTEGER,
                cost_per_input_token DOUBLE,
                output_token INTEGER,
                cost_per_output_token DOUBLE,
                cost DOUBLE,
                insert_time INTEGER
            );
            """,
            commit=True,
            on_raise=True,
        )
    
    def drop_tables(self):
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            DROP TABLE IF EXISTS audit_log;
            """,
            commit=True,
            on_raise=True,
        )
        cursor.close()

        for key in self.status_cache.keys():
            del self.status_cache[key]
    
    def build_audit_status_key(self, user_id: int, billing_period: str) -> str:
        return f"audit:{user_id}:{billing_period}"

    def insert_audit_log(self, user_id: int, billing_period: str, thread_id: str, model: str, input_token: int, output_token: int) -> None:
        # get current pricing
        cost_per_input_token, cost_per_output_token = self.pricing_model.get_current_pricing_by_model(model)
        cost = input_token * cost_per_input_token + output_token * cost_per_output_token
        now = int(time.time())
        
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO audit_log (user_id, billing_period, thread_id, model, input_token, cost_per_input_token, output_token, cost_per_output_token, cost, insert_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (user_id, billing_period, thread_id, model, input_token, cost_per_input_token, output_token, cost_per_output_token, cost, now),
            commit=True,
            on_raise=True,
        )
        cursor.close()

        key = self.build_audit_status_key(user_id, billing_period)
        self.status_cache.setdefault(key, '0')
        self.status_cache[key] = str(float(self.status_cache[key]) + cost)

    def get_total_cost_by_user_id_billing_period(self, user_id: int, billing_period: str) -> float:
        key = self.build_audit_status_key(user_id, billing_period)
        if key in self.status_cache:
            return float(self.status_cache[key])
        else:
            return self.fetchone(
                """
                SELECT SUM(cost) FROM audit_log WHERE user_id = ? AND billing_period = ?;
                """,
                (user_id, billing_period),
            )

if __name__ == "__main__":
    # unittest
    conn = sqlite3.connect("unittest.db")
    audit_model = AuditModel(conn)

    audit_model.drop_tables()
    audit_model.create_tables()

    # prepare data
    user_id = 1
    billing_period = "2021-01"
    model = "gpt2"
    thread_id = "thread-1"
    audit_model.pricing_model.create_pricing(model, 0.01, 0.02, 0)

    # test insert
    audit_model.insert_audit_log(user_id, billing_period, thread_id, model, 100, 100)
    res = audit_model.get_total_cost_by_user_id_billing_period(user_id, billing_period)
    print(res)

    # test insert
    audit_model.insert_audit_log(user_id, billing_period, thread_id, model, 200, 200)
    res = audit_model.get_total_cost_by_user_id_billing_period(user_id, billing_period)
    print(res)
        