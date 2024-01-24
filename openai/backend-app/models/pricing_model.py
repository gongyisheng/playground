import sqlite3
from typing import Tuple

from base_model import BaseModel

class PricingModel(BaseModel):

    def __init__(self, conn: sqlite3.Connection) -> None:
        super().__init__(conn)

    def create_tables(self) -> None:
        # pricing table
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            CREATE TABLE IF NOT EXISTS pricing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model VARCHAR(255),
                cost_per_input_token DOUBLE,
                cost_per_output_token DOUBLE,
                start_time INTEGER,
                end_time INTEGER
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
            DROP TABLE IF EXISTS pricing;
            """,
            commit=True,
            on_raise=True,
        )
        cursor.close()
    
    def create_pricing(self, model: str, cost_per_input_token: float, cost_per_output_token: float, start_time: int, end_time: int = -1) -> None:
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            INSERT INTO pricing (model, cost_per_input_token, cost_per_output_token, start_time, end_time)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                model,
                cost_per_input_token,
                cost_per_output_token,
                start_time,
                end_time,
            ),
            commit=True,
            on_raise=True,
        )
        cursor.close()

    def update_pricing(self, model: str, cost_per_input_token: float, cost_per_output_token: float, start_time: int, end_time: int = -1) -> None:
        # set current pricing end_time of the model to start_time of the new pricing
        cursor = self.conn.cursor()
        self._execute_sql(
            cursor,
            """
            UPDATE pricing SET end_time = ?
            WHERE model = ? AND end_time = -1;
            """,
            (
                start_time,
                model,
            ),
            commit=True,
            on_raise=True,
        )
        cursor.close()

        # create new pricing
        self.create_pricing(model, cost_per_input_token, cost_per_output_token, start_time, end_time)
    
    def get_pricing_by_model(self, model: str) -> list:
        return self.fetchall(
            """
            SELECT cost_per_input_token, cost_per_output_token, start_time, end_time FROM pricing WHERE model = ?;
            """,
            (model,),
            on_raise=True,
        )
    
    def get_current_pricing_by_model(self, model: str) -> Tuple:
        return self.fetchone(
            """
            SELECT cost_per_input_token, cost_per_output_token FROM pricing WHERE model = ? AND end_time = -1;
            """,
            (model,),
            on_raise=True,
        )

if __name__ == '__main__':
    conn = sqlite3.connect('unittest.db')
    pricing_model = PricingModel(conn)

    pricing_model.drop_tables()
    pricing_model.create_tables()

    model = 'davinci'
    cost_per_input_token = 0.0001
    cost_per_output_token = 0.0002
    start_time = 1620000000
    end_time = -1

    pricing_model.create_pricing(model, cost_per_input_token, cost_per_output_token, start_time, end_time)
    pricing = pricing_model.get_current_pricing_by_model(model)
    print("current pricing: ", pricing)

    model = 'davinci'
    cost_per_input_token = 1
    cost_per_output_token = 2
    start_time = 1630000000
    pricing_model.update_pricing(model, cost_per_input_token, cost_per_output_token, start_time, end_time)
    pricing = pricing_model.get_current_pricing_by_model(model)
    print("current pricing: ", pricing)

    pricing = pricing_model.get_pricing_by_model(model)
    print("all pricing: ", pricing)
    