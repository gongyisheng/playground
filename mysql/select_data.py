import sys

import mysql.connector

# Define number of rows to insert
try:
    num_rows = int(sys.argv[1])
except Exception:
    num_rows = 100

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
    cursor.execute("SELECT * FROM user_age WHERE id=530")
    rows = cursor.fetchall()
    print(f"SELECT row {i}")

# close connection
conn.close()

print("Done selecting data")
