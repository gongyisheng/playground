### PostgreSQL Meta-Commands (psql)
These commands are used inside the PostgreSQL command line utility `psql`:

#### 1. List all databases
```
\l
```

#### 2. List all tables in the current database
```
\dt
```

#### 3. Get description of a table structure
```
\d tablename
```

#### 4. Connect to another database
```
\c dbname
```

#### 5. Display query results in a pretty-format
```
\x
```

#### 6. Quit psql
```
\q
```

### Basic SQL Queries

#### 1. Create a Database
```sql
CREATE DATABASE mydatabase;
```

#### 2. Drop a Database
```sql
DROP DATABASE mydatabase;
```

#### 3. Connect to a Database
```
\c mydatabase
```

#### 4. Create a Table
```sql
CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    date_of_birth DATE,
    salary DECIMAL
);
```

#### 5. Drop a Table
```sql
DROP TABLE tablename;
```

#### 6. Insert Data
```sql
INSERT INTO employees (first_name, last_name, email, date_of_birth, salary)
VALUES ('John', 'Doe', 'john.doe@example.com', '1980-01-01', 50000);
```

#### 9. Update Data
```sql
UPDATE employees SET salary = 52000 WHERE id = 1;
```

#### 10. Delete Data
```sql
DELETE FROM employees WHERE id = 1;
```

#### 11. Add a Column
```sql
ALTER TABLE employees ADD COLUMN department VARCHAR(50);
```

#### 12. Drop a Column
```sql
ALTER TABLE employees DROP COLUMN department;
```
