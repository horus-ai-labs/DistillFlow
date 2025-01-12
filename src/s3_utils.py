import os
import boto3
from datetime import datetime


def upload_to_s3(bucket_name, dir_path='src/results', access_key=None, secret_key=None, region=None):
    """
    Upload a file to an S3 bucket with flexible credential handling.

    :param bucket_name: Bucket to upload to
    :param dir_path: Directory path containing files to upload
    :param access_key: Optional AWS access key. If None, will check S3_ACCESS_KEY env var
    :param secret_key: Optional AWS secret key. If None, will check S3_SECRET_KEY env var
    :param region: Optional AWS region. If None, will check AWS_REGION env var or default to us-west-2
    :return: True if file was uploaded, else False
    """
    # Handle credentials with parameter priority over environment variables
    aws_access_key = access_key or os.getenv('S3_ACCESS_KEY')
    aws_secret_key = secret_key or os.getenv('S3_SECRET_KEY')
    aws_region = region or os.getenv('AWS_REGION', 'us-west-2')

    # Validate credentials
    if not aws_access_key or not aws_secret_key:
        raise ValueError("AWS credentials not found. Please provide them as parameters or set S3_ACCESS_KEY and S3_SECRET_KEY environment variables.")

    # Set credentials in environment if they came from parameters
    if access_key:
        os.environ['S3_ACCESS_KEY'] = access_key
    if secret_key:
        os.environ['S3_SECRET_KEY'] = secret_key
    if region:
        os.environ['AWS_REGION'] = region

    # Validate credentials
    if not aws_access_key or not aws_secret_key:
        EnvironmentError("AWS credentials not found in environment variables.")
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
                raise Exception("Cannot upload file")


