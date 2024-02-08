import sqlite3
import sys

# add path
sys.path.append("../")
from models.pricing_model import PricingModel
db_file = sys.argv[1] if len(sys.argv) > 1 else "unittest.db"

conn = sqlite3.connect(db_file)
pricing_model = PricingModel(conn)

if pricing_model.get_current_pricing_by_model("gpt-3.5-turbo-1106") is None:
    pricing_model.create_pricing(
        "gpt-3.5-turbo-1106", 0.001 / 1000, 0.002 / 1000, 0
    )
if pricing_model.get_current_pricing_by_model("gpt-4-0125-preview") is None:
    pricing_model.create_pricing(
        "gpt-4-0125-preview", 0.01 / 1000, 0.03 / 1000, 0
    )
if pricing_model.get_current_pricing_by_model("gpt-3.5-turbo-0125") is None:
    pricing_model.create_pricing(
        "gpt-3.5-turbo-0125", 0.0005 / 1000, 0.0015 / 1000, 0
    )
