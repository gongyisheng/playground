# Common databricks SQL

# History & Time travel
```
## History
DESCRIBE HISTORY <table_name>;

## Query on a table of history version
SELECT * FROM <table_name> VERSION AS OF <version>;

## Restore table to a history verison
RESTORE TABLE <table_name> TO VERSION AS OF <version>;
```