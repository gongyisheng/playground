import os
import time
import boto3
from botocore.exceptions import ClientError

# Configuration
MINIO_URL = "http://127.0.0.1:9000"  # or 'http://127.0.0.1:9000'
MINIO_ACCESS_KEY = "root"
MINIO_SECRET_KEY = "minioroot"
BUCKET_NAME = "test"
OBJECT_NAME = "test-object.txt"
DOWNLOAD_FILE = "downloaded_test-object.txt"

# Create a session
session = boto3.session.Session()
s3_client = session.client(
    service_name='s3',
    endpoint_url=MINIO_URL,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

def main():
    # Create bucket if it doesn't exist
    try:
        s3_client.head_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' already exists.")
    except ClientError:
        s3_client.create_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket '{BUCKET_NAME}' created.")

    # Create a test file
    with open(OBJECT_NAME, 'w') as file:
        file.write("Hello, MinIO!")  # Create a sample file

    # Upload the file to MinIO
    try:
        s3_client.upload_file(OBJECT_NAME, BUCKET_NAME, OBJECT_NAME)
        print(f"Object '{OBJECT_NAME}' uploaded to bucket '{BUCKET_NAME}'.")
    except ClientError as e:
        print("Error occurred while uploading:", e)

    # Download the object
    try:
        s3_client.download_file(BUCKET_NAME, OBJECT_NAME, DOWNLOAD_FILE)
        print(f"Object '{OBJECT_NAME}' downloaded from bucket '{BUCKET_NAME}' as '{DOWNLOAD_FILE}'.")
    except ClientError as e:
        print("Error occurred while downloading:", e)

    # Clean up
    os.remove(OBJECT_NAME)  # Remove the original file
    os.remove(DOWNLOAD_FILE)  # Remove the downloaded file
    try:
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=OBJECT_NAME)
        print(f"Object '{OBJECT_NAME}' removed from bucket '{BUCKET_NAME}'.")
    except ClientError as e:
        print("Error occurred while removing:", e)

if __name__ == "__main__":
    main()