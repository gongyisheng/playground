# this script does not work
import boto3
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import TimestampType, DoubleType, StringType, NestedField

# MinIO configuration
MINIO_ENDPOINT = 'http://127.0.0.1:9000'  # MinIO endpoint
ACCESS_KEY = 'test'  # MinIO access key
SECRET_KEY = 'abc'  # MinIO secret key
BUCKET_NAME = 'minecraft-back-up'  # Your MinIO bucket
TABLE_NAME = 'test.test'  # Table name

schema = Schema(
    NestedField(field_id=1, name="key", field_type=StringType(), required=False),
    NestedField(field_id=2, name="value", field_type=StringType(), required=False),
)

# Initialize Boto3 S3 client for MinIO
s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

# Initialize Iceberg S3 catalog for MinIO
s3_catalog = load_catalog("test", client=s3_client, uri="s3://minecraft-back-up", type="GLUE")

s3_catalog.create_table(
    identifier="test.test",
    schema=schema,
    location="s3://minecraft-back-up/test",
)

# Load the table
table = s3_catalog.load_table(TABLE_NAME)

def perform_acid_operations():
    # Start a transaction using Iceberg
    with table.new_transaction() as tx:
        # Example of an INSERT operation
        print("Inserting new records...")
        tx.insert([
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25}
        ])
        
        # Example of an UPDATE operation
        print("Updating records...")
        tx.update()
        tx.where("id = 1").set("age", 31)
        
        # Example of a DELETE operation
        print("Deleting a record...")
        tx.delete()
        tx.where("id = 2")
        
        # Commit the transaction
        print("Committing the transaction...")
        tx.commit()

if __name__ == "__main__":
    perform_acid_operations()