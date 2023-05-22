import sys
import time

import mysql.connector

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
    "30m": 30_000_000,
    "50m": 50_000_000,
    "80m": 80_000_000
}

table_size = sys.argv[1]
if table_size not in supported_table:
    print(f"Invalid table size. supported table: {supported_table.keys()}")
    raise Exception
round = int(sys.argv[2])
sql_id = int(sys.argv[3])

table = f"{table_size}_row_test"
print(f"Test Job: count * test. Table: {table}")

sql = {
    1: f"select count(*) from {table}",
    2: f"select count(*) from {table} where id = 12345",
    3: f"select count(*) from {table} where insert_time = 12345",
    4: f"select * from {table} where insert_time = 12345"
}

# Connect to MySQL server
conn = mysql.connector.connect(**db_cred)

# Create cursor
cursor = conn.cursor()
total_time = 0
cursor.execute("show status like 'innodb_buffer_pool_pages_data'")
start_status = cursor.fetchall()

for i in range(round):
    start = time.perf_counter()
    cursor.execute(sql[sql_id])
    data = cursor.fetchall()
    end = time.perf_counter()
    print(f"select_time: {end-start}, round: {i}")
    total_time += end-start
cursor.execute("show status like 'innodb_buffer_pool_pages_data'")
end_status = cursor.fetchall()
print(f"Total_round: {round}, avg_time: {total_time/round}")
print(f"start_status: {start_status}, end_status: {end_status}")

# Commit changes and close connection
conn.commit()
conn.close()

print("Done selecting data")
