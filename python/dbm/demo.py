import dbm

# Open database, creating it if necessary.
db = dbm.open('cache', 'c')

# Record some values
db['www.python.org'] = 'Python Website'
db['www.cnn.com'] = 'Cable News Network'

# Note that the keys are considered bytes now.
assert db[b'www.python.org'] == b'Python Website'

# Close when done.
db.close()

# Reopen the database.
db = dbm.open('cache', 'r')
print(db)