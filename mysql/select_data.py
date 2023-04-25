import sys

import mysql.connector

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

table = f"{payload_size}{unit}_test"
print(f"Test Job: insert {num_rows} rows of {payload_size}{unit} data. Table: {table}")

# Define connection parameters
config = {
    'user': '',
    'password': '',
    'host': 'test-database-1.cwib8arbo5fd.us-east-1.rds.amazonaws.com',
    'database': '',
}

# Connect to MySQL server
conn = mysql.connector.connect(**config)

# Create cursor
cursor = conn.cursor()

for i in range(num_rows):
    cursor.execute(f"SELECT * FROM {table} WHERE id={i}")
    rows = cursor.fetchall()
    print(f"SELECT row {i}")

# close connection
conn.close()

print("Done selecting data")
