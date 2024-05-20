import json
import shutil
import boto3
from botocore.exceptions import NoCredentialsError


def zip_folder(folder_path, output_zip):
    """Compress the folder into a zip file."""
    shutil.make_archive(output_zip, "zip", folder_path)


def upload_to_s3(file_path, bucket_name, s3_file_name):
    """Upload a file to an S3 bucket."""
    s3 = boto3.client(
        "s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )
    try:
        s3.upload_file(
            file_path,
            bucket_name,
            s3_file_name,
            ExtraArgs={"StorageClass": storage_class},
        )
        print(f"File {s3_file_name} uploaded to S3 bucket {bucket_name}.")
    except FileNotFoundError:
        print("The file was not found.")
    except NoCredentialsError:
        print("Credentials not available.")


if __name__ == "__main__":
    from argparse import ArgumentParser

    options = ArgumentParser()
    options.add_argument(
        "-p", "--path", dest="path", required=True, help="path to the folder to backup"
    )
    options.add_argument(
        "-skip-zip",
        "--skip-zip",
        dest="skip_zip",
        required=False,
        help="skip zip the folder",
        action="store_true",
    )
    options.add_argument(
        "-skip-upload",
        "--skip-upload",
        dest="skip_upload",
        required=False,
        help="skip upload the zip file",
        action="store_true",
    )
    args = options.parse_args()

    base_folder = "/media/usbdisk"
    key_file_path = f"{base_folder}/codebase/user-key/s3/key.json"
    subpath = args.path

    folder_path = f"{base_folder}/{subpath}"  # Replace with the path to your folder
    output_zip = f"{base_folder}/backup/{subpath}"  # Replace with your desired output zip file name (without .zip extension)
    print(f"backup path: {folder_path}, local zip path: {output_zip}.zip")

    bucket_name = "yisheng-backup"  # Replace with your S3 bucket name
    s3_file_name = f"{subpath}.zip"  # Replace with desired path and file name in S3
    storage_class = "DEEP_ARCHIVE"
    print(
        f"bucket name: {bucket_name}, s3 file name: {s3_file_name}, storage class: {storage_class}"
    )

    with open(key_file_path, "r", encoding="utf_8") as f:
        content = f.read()
        content = json.loads(content)
        access_key = content["access_key"]
        secret_key = content["secret_key"]

    # Compress the folder
    if not args.skip_zip:
        zip_folder(folder_path, output_zip)
        print("Zip file created.")

    # Upload to S3
    if not args.skip_upload:
        upload_to_s3(f"{output_zip}.zip", bucket_name, s3_file_name)
        print("Backup uploaded to S3.")
