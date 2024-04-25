import sqlite3
import sys

# add path
sys.path.append("../")
from models.pricing_model import PricingModel


def init_price(db_conn: sqlite3.Connection):
    pricing_model = PricingModel(db_conn)

    if pricing_model.get_current_pricing_by_model("gpt-3.5-turbo-0125") is None:
        pricing_model.create_pricing(
            "gpt-3.5-turbo-0125", 0.0005 / 1000, 0.0015 / 1000, 0
        )
    if pricing_model.get_current_pricing_by_model("gpt-4-turbo-2024-04-09") is None:
        pricing_model.create_pricing(
            "gpt-4-turbo-2024-04-09", 10 / 1000000, 30 / 1000000, 0
        )
    if pricing_model.get_current_pricing_by_model("llama3-70b-8192") is None:
        pricing_model.create_pricing("llama3-70b-8192", 0, 0, 0)
    if pricing_model.get_current_pricing_by_model("claude-3-opus-20240229") is None:
        pricing_model.create_pricing(
            "claude-3-opus-20240229", 15 / 1000000, 75 / 1000000, 0
        )


if __name__ == "__main__":
    db_file = sys.argv[1] if len(sys.argv) > 1 else "unittest.db"
    db_conn = sqlite3.connect(db_file)
    init_price(db_conn)
