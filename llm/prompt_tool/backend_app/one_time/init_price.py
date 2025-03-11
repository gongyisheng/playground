import sqlite3
import sys

# add path
sys.path.append("../")
from models.pricing_model import PricingModel


def init_price(db_conn: sqlite3.Connection):
    pricing_model = PricingModel(db_conn)

    if pricing_model.get_current_pricing_by_model("gpt-4o-mini-2024-07-18") is None:
        pricing_model.create_pricing(
            "gpt-4o-mini-2024-07-18", 0.15 / 1000000, 0.6 / 1000000, 0
        )
    if pricing_model.get_current_pricing_by_model("gpt-4o-2024-11-20") is None:
        pricing_model.create_pricing("gpt-4o-2024-11-20", 2.5 / 1000000, 10 / 1000000, 0)
    if pricing_model.get_current_pricing_by_model("llama-3.2-90b-vision-preview") is None:
        pricing_model.create_pricing("llama-3.2-90b-vision-preview", 0, 0, 0)
    if pricing_model.get_current_pricing_by_model("claude-3-5-sonnet-20241022") is None:
        pricing_model.create_pricing(
            "claude-3-7-sonnet-20250219", 3 / 1000000, 15 / 1000000, 0
        )
    if pricing_model.get_current_pricing_by_model("o3-mini-2025-01-31") is None:
        pricing_model.create_pricing("o3-mini-2025-01-31", 1.1 / 1000000, 4.4 / 1000000, 0)
    if pricing_model.get_current_pricing_by_model("deepseek-ai/DeepSeek-R1") is None:
        pricing_model.create_pricing("deepseek-ai/DeepSeek-R1", 4 / 7.25 / 1000000, 16 / 7.25 / 1000000, 0)
    if pricing_model.get_current_pricing_by_model("deepseek-ai/DeepSeek-V3") is None:
        pricing_model.create_pricing("deepseek-ai/DeepSeek-V3", 2 / 7.25 / 1000000, 8 / 7.25 / 1000000, 0)


if __name__ == "__main__":
    db_file = sys.argv[1] if len(sys.argv) > 1 else "unittest.db"
    db_conn = sqlite3.connect(db_file)
    init_price(db_conn)
