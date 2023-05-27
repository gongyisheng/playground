import sys
import time

import mysql.connector

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

table_size = sys.argv[1]
if table_size not in supported_table:
    print(f"Invalid table size. supported table: {supported_table.keys()}")
    raise Exception
round = int(sys.argv[2])
sql_id = int(sys.argv[3])

table = f"{table_size}_big_pk_test"
print(f"Test Job: count * test. Table: {table}")
target_key = "OawrGDEUPtVxudqspDncYswuJReaLenIRaazDUrNuvoJOfpTfrJOfnJWQZZuAklIMSKytNsiIOPBXvmLlanDTeBVaxEkeilJdnOFpMcXGQLesGYahIUpbuwpkloZneWSvUrCbfGhzAWFqrtxPoITZgdZvTrGFGrYMDRHtzDIrazFVZOFHrZxylgEwNRoHdvwDIXeWuJxDpaCsqSjgQDhEjIMumVPtySerRLnlQjbVtlIJoqqslwZwRZnIbFCHaCFMeSwOYXKPCjZiQfHucMfgpZmVJXJzbEKxnSbOrBwzeXwWYNCvjWgiIKKXLEygZlncrKfEPuwqBcsSYmvPQjwsufvVTkHUVrGcAEvxCUziRvXtcOCfQHcnlaXcBdNfTcQMXwCcUeUgVpohdxaagcjAGgybOgjQPMFraipkrBgcVkOQjXVzaRyQTZvksmUxTPKuMYhYHSgdYvtSwhbKUEQiURkTBuZqfrjcicTrovimVdLrcBhbgqjLIFkYsogfrSSBhDOAHImynXMxNjeFtNVbWFxlJTZycWxkyxMomphFzFVguzYbmcYiRgNyuPfBOXdgemLcLgCTcCXiEOARhmOEGwERFQyLvwCVrwGNVXGHAwVPrktEqjZQXkjcNvXmvjOcJnrDHwHLCVTWIJrReGvjByLjVWrGLdlTvCquSxuzIwyRzPScirvYiodNNjROjRmFvwvvxClCmlhoXigVdQQOejVcHrEeBhtbLnLCgezWGgROxWVfHcVENbDWFSvnFJI"

sql = {
    1: f"select * from {table} where big_pk = \"{target_key}\"",
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
