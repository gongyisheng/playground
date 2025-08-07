import sqlite3
import sys

# add path
sys.path.append("../")
from models.pricing_model import PricingModel


def init_price(db_conn: sqlite3.Connection):
    pricing_model = PricingModel(db_conn)

    if pricing_model.get_current_pricing_by_model("gpt-5-2025-08-07") is None:
        pricing_model.create_pricing("gpt-5-2025-08-07", 1.25 / 1000000, 10 / 1000000, 0)
    if pricing_model.get_current_pricing_by_model("gpt-5-mini-2025-08-07") is None:
        pricing_model.create_pricing("gpt-5-2025-08-07", 0.25 / 1000000, 2 / 1000000, 0)
    # if pricing_model.get_current_pricing_by_model("llama-3.2-90b-vision-preview") is None:
    #     pricing_model.create_pricing("llama-3.2-90b-vision-preview", 0, 0, 0)
    if pricing_model.get_current_pricing_by_model("claude-sonnet-4-20250514") is None:
        pricing_model.create_pricing("claude-sonnet-4-20250514", 3 / 1000000, 15 / 1000000, 0)
    if pricing_model.get_current_pricing_by_model("o4-mini-2025-04-16") is None:
        pricing_model.create_pricing("o4-mini-2025-04-16", 0.55 / 1000000, 2.2 / 1000000, 0)
    if pricing_model.get_current_pricing_by_model("google/gemini-2.5-pro") is None:
        pricing_model.create_pricing("google/gemini-2.5-pro", 2.5 / 1000000, 15 / 1000000, 0)



if __name__ == "__main__":
    db_file = sys.argv[1] if len(sys.argv) > 1 else "unittest.db"
    db_conn = sqlite3.connect(db_file)
    init_price(db_conn)
