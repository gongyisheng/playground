import random
import sys

import mysql.connector

from config import db_cred

try:
    payload_size = int(sys.argv[1])
    unit = str(sys.argv[2])
    if unit not in ["b","kb","mb"]:
        print("Invalid unit. Please use b, kb, or mb")
        raise Exception
except Exception:
    payload_size = 1
    unit = "mb"

# Define number of rows to insert
try:
    num_rows = int(sys.argv[3])
except Exception:
    num_rows = 100

try:
    round = int(sys.argv[4])
except Exception:
    round = 1

table = f"{payload_size}{unit}_test"
print(f"Test Job: selects {num_rows} rows of {payload_size}{unit} data. Table: {table}, Round: {round}")

# Connect to MySQL server
conn = mysql.connector.connect(**db_cred)

# Create cursor
cursor = conn.cursor()

for _ in range(round):
    for i in range(num_rows):
        rand_id = random.randint(0, num_rows)
        cursor.execute(f"SELECT * FROM {table} WHERE id={rand_id}")
        rows = cursor.fetchall()
        print(f"SELECT row {i}")

# close connection
conn.close()

print("Done selecting data")
