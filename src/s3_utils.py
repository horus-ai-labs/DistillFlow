import os
import boto3
from datetime import datetime


def upload_to_s3(bucket_name, dir_path='src/results'):
    """
    Upload a file to an S3 bucket.

    :param bucket_name: Bucket to upload to
    :return: True if file was uploaded, else False
    """
    # Get environment variables for AWS credentials
    aws_access_key = os.getenv('S3_ACCESS_KEY')
    aws_secret_key = os.getenv('S3_SECRET_KEY')
    aws_region = os.getenv('AWS_REGION', 'us-west-2')

    # Validate credentials
    if not aws_access_key or not aws_secret_key:
        raise EnvironmentError("AWS credentials not found in environment variables.")
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )
    timestamp = datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')
    s3_folder_prefix = f"{timestamp}/"

    # Walk through the directory and upload each file
    for root, _, files in os.walk(dir_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            # Generate the relative path to maintain folder structure in S3
            relative_path = os.path.relpath(file_path, dir_path)
            s3_key = f"{s3_folder_prefix}{relative_path}"

            try:
                # Upload the file
                s3_client.upload_file(file_path, bucket_name, s3_key)
                print(f"Uploaded: {file_path} -> s3://{bucket_name}/{s3_key}")
            except Exception as e:
                print(f"An error occurred: {e}")
                return False
                # # Use boto3 client
    # try:
    #     s3_client = boto3.client(
    #         's3',
    #         aws_access_key_id=aws_access_key,
    #         aws_secret_access_key=aws_secret_key,
    #         region_name=aws_region
    #     )
    #
    #     if object_name is None:
    #         object_name = file_name
    #
    #     # Upload file
    #     s3_client.upload_file(file_name, bucket_name, object_name)
    #     print(f"File {file_name} uploaded to {bucket_name}/{object_name}")
    #     return True

    # except FileNotFoundError:
    #     print(f"Error: The file {file_name} was not found.")
    #     return False
    # except NoCredentialsError:
    #     print("Error: No AWS credentials found.")
    #     return False
    # except PartialCredentialsError:
    #     print("Error: Incomplete AWS credentials.")
    #     return False
