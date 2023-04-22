import MySQLdb
import random
import string

# Define connection parameters
config = {
    'user': '',
    'password': '',
    'host': 'test-database-1.cwib8arbo5fd.us-east-1.rds.amazonaws.com',
    'database': '',
}

# Connect to MySQL server
conn = MySQLdb.connect(**config)

# Create cursor
cursor = conn.cursor()

# Define function to generate random strings
def random_string(length):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))

# Define function to generate random integers
def random_int(length):
    return random.randint(0, 10**length)

# Define function to generate random user data
def generate_user_age():
    return (random_string(10), random_int(2))

# Define number of rows to insert
num_rows = 100

# Insert random data into user table
for i in range(num_rows):
    user_data = generate_user_age()
    cursor.execute("INSERT INTO user_age (user_id, age) VALUES (%s, %s)", user_data)

# Commit changes and close connection
conn.commit()
conn.close()
