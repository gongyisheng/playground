import pyarrow as pa
import pyarrow.parquet as pq

data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NY', 'LA', 'SF']
}

table = pa.table(data)

pq.write_table(table, 'data.parquet')

