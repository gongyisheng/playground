import sys

import mysql.connector
import random
import string

from config import db_cred

supported_table = {
    "5": 5,
    "25": 25,
    "125": 125,
    "625": 625,
    "3125": 3125,
    "15625": 15625,
    "78125": 78125,
    "390625": 390625,
}

table_size = str(sys.argv[1])
if table_size not in supported_table:
    print(f"Invalid table size. supported table: {supported_table.keys()}")
    raise Exception
num_rows = int(table_size)

table = f"{table_size}_big_pk_test"
print(f"Test Job: insert {num_rows} rows of data. Table: {table}")

# Connect to MySQL server
conn = mysql.connector.connect(**db_cred)

# Create cursor
cursor = conn.cursor()

def random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

def compact_string(string):
    return "\"" + string + "\""

fixed_tail = random_string(756)

for i in range(num_rows):
    head = random_string(12)
    pk = compact_string(head + fixed_tail)
    sql = f"INSERT INTO {table} (big_pk, big_index_key) VALUES ({pk}, {pk})"
    cursor.execute(sql)
    if i%100 == 0:
        conn.commit()
        print(f"Inserted row {i}")

# Commit changes and close connection
conn.commit()
conn.close()

print("Done inserting data")
