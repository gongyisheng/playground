import sys

import mysql.connector
import random
import string

from config import db_cred

supported_table = {
    "100k": 100_000, 
    "200k": 200_000, 
    "500k": 500_000, 
    "1m": 1_000_000, 
    "2m": 2_000_000, 
    "5m": 5_000_000,
    "10m": 10_000_000, 
    "20m": 20_000_000, 
    "50m": 50_000_000
}

table_size = sys.argv[1]
if table_size not in supported_table:
    print(f"Invalid table size. supported table: {supported_table.keys()}")
    raise Exception
num_rows = int(sys.argv[2])


table = f"{table_size}_row_test"
print(f"Test Job: insert {num_rows} rows of data. Table: {table}")
max_rows = supported_table[table_size]

# Connect to MySQL server
conn = mysql.connector.connect(**db_cred)

# Create cursor
cursor = conn.cursor()

def random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

def compact_string(string):
    return "\"" + string + "\""

# Define function to generate random integers
def random_int(_min, _max):
    return random.randint(_min, _max)

for i in range(num_rows):
    person_id = random_int(0, max_rows)
    person_name = compact_string(random_string(200))
    insert_time = random_int(0, 1684298424)
    update_time = random_int(insert_time, 1684298424)
    sql = f"INSERT INTO {table} (person_id, person_name, insert_time, update_time) VALUES ({person_id}, {person_name}, {insert_time}, {update_time})"
    cursor.execute(sql)
    if i%100 == 0:
        conn.commit()
        print(f"Inserted row {i}")

# Commit changes and close connection
conn.commit()
conn.close()

print("Done inserting data")
