import argparse
import os
import subprocess
import time
from datetime import datetime

from google.cloud import compute_v1
from google.oauth2 import service_account
from google.auth import default
import getpass

current_user = getpass.getuser()


def get_credentials(service_account_path=None):
    """Get credentials either from service account or application default."""
    try:
        if service_account_path:
            credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        else:
            credentials, project = default()
        return credentials
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        print("\nPlease ensure you have either:")
        print("1. Run 'gcloud auth application-default login'")
        print("2. Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
        raise


def get_git_config():
    """Get git user.name and user.email from local machine."""
    try:
        git_email = subprocess.check_output(['git', 'config', '--get', 'user.email']).decode().strip()
        git_name = subprocess.check_output(['git', 'config', '--get', 'user.name']).decode().strip()
        return git_email, git_name
    except subprocess.CalledProcessError:
        return None, None


def create_instance(project_id, zone, instance_name, machine_type, gpu_count,
                    boot_disk_size, data_disk_size, env_vars, credentials=None):
    """Create a VM instance with GPU on Google Cloud Platform."""
    instance_client = compute_v1.InstancesClient(credentials=credentials)

    # Configure the machine
    machine_type_path = f"zones/{zone}/machineTypes/{machine_type}"

    # Configure the boot disk
    boot_disk = compute_v1.AttachedDisk(
        auto_delete=True,
        boot=True,
        initialize_params=compute_v1.AttachedDiskInitializeParams(
            disk_size_gb=boot_disk_size,
            disk_type=f"zones/{zone}/diskTypes/pd-standard",
            source_image=f"projects/{project_id}/global/images/py12-cu124-20250111"
            # source_image="projects/cos-cloud/global/images/family/cos-stable"
            # source_image="projects/ml-images/global/images/c0-deeplearning-common-cu124-v20241118-debian-11-py310"
        ),
    )

    # Configure additional data disk
    data_disk = compute_v1.AttachedDisk(
        auto_delete=True,
        boot=False,
        initialize_params=compute_v1.AttachedDiskInitializeParams(
            disk_size_gb=data_disk_size,
            disk_type=f"zones/{zone}/diskTypes/pd-ssd",
        ),
    )

    # Configure GPU
    gpu_config = compute_v1.AcceleratorConfig(
        accelerator_count=gpu_count,
        accelerator_type=f"zones/{zone}/acceleratorTypes/nvidia-tesla-a100"
    )

    # Configure networking
    network_interface = compute_v1.NetworkInterface(
        network="global/networks/default",
        access_configs=[compute_v1.AccessConfig(name="External NAT")]
    )

    sa = compute_v1.ServiceAccount()
    sa.email = "distillflow-vm@distillation-101.iam.gserviceaccount.com"
    sa.scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/monitoring.write",
        "https://www.googleapis.com/auth/service.management.readonly",
        "https://www.googleapis.com/auth/servicecontrol",
        "https://www.googleapis.com/auth/trace.append"
    ]

    # Get git config from local machine
    git_email, git_name = get_git_config()

    # Read both public and private SSH keys
    ssh_pub_key_path = os.path.expanduser('~/.ssh/id_ed25519.pub')
    ssh_private_key_path = os.path.expanduser('~/.ssh/id_ed25519')

    try:
        with open(ssh_pub_key_path, 'r') as f:
            pub_key = f.read().strip()
        with open(ssh_private_key_path, 'r') as f:
            private_key = f.read()
    except FileNotFoundError:
        raise Exception("SSH keys not found. Generate them using 'ssh-keygen -t ed25519'")

    aws_access_key = os.getenv('S3_ACCESS_KEY', None)
    aws_secret_key = os.getenv('S3_SECRET_KEY', None)
    aws_region = "us-west-2"

    if aws_access_key is None or aws_secret_key is None:
        raise Exception("S3_ACCESS_KEY and S3_SECRET_KEY not set on your machine, please set it.")

    startup_script = f"""#!/bin/bash
exec 1>/tmp/startup-log.txt 2>&1  # Redirect all output to a log file
set -x  # Print commands as they execute

# Create .aws directory
mkdir -p /home/{current_user}/.aws

# Create credentials file
cat > /home/{current_user}/.aws/credentials << EOL
[default]
aws_access_key_id = {aws_access_key}
aws_secret_access_key = {aws_secret_key}
EOL

# Create config file
cat > /home/{current_user}/.aws/config << EOL
[default]
region = {aws_region}
EOL

# Set proper ownership
chown -R {current_user}:{current_user} /home/{current_user}/.aws
chmod 600 /home/{current_user}/.aws/credentials
chmod 600 /home/{current_user}/.aws/config

chmod 700 /home/{current_user}/.ssh

# Create id_ed25519 private key
cat > /home/{current_user}/.ssh/id_ed25519 << 'EOL'
{private_key}
EOL
chmod 600 /home/{current_user}/.ssh/id_ed25519

# Create id_ed25519.pub public key
cat > /home/{current_user}/.ssh/id_ed25519.pub << 'EOL'
{pub_key}
EOL
chmod 644 /home/{current_user}/.ssh/id_ed25519.pub

# Set proper ownership
chown -R {current_user}:{current_user} /home/{current_user}/.ssh

# Set known hosts
{f'su - {current_user} -c \'ssh-keyscan -t rsa github.com >> /home/{current_user}/.ssh/known_hosts\''}
chmod 644 /home/{current_user}/.ssh/known_hosts

# Set git config if available
{f'su - {current_user} -c \'git config --global user.email "{git_email}"\'' if git_email else ''}
{f'su - {current_user} -c \'git config --global user.name "{git_name}"\'' if git_name else ''}
{f'su - {current_user} -c \'git config --global --add safe.directory /app\''}
{f'su - {current_user} -c \'git config --global pull.rebase false\''}

chown -R ${current_user}:${current_user} /app
{f'su - {current_user} -c \'cd /app && git remote set-url origin git@github.com:horus-ai-labs/DistillFlow.git && git pull\''}

cat >> /app/venv/bin/activate << EOL

# Add env vars to .bashrc
export S3_ACCESS_KEY={aws_access_key}
export S3_SECRET_KEY={aws_secret_key}
export AWS_REGION={aws_region}
EOL
chown ${current_user}:${current_user} /app/venv/bin/activate

source /app/venv/bin/activate

echo "Startup script completed" >> /tmp/startup-log.txt
    """

    # Create the instance configuration
    instance = compute_v1.Instance(
                name=instance_name,
                machine_type=machine_type_path,
                disks=[boot_disk, data_disk],
                network_interfaces=[network_interface],
                guest_accelerators=[gpu_config],
                scheduling=compute_v1.Scheduling(
                    on_host_maintenance="TERMINATE",
                    automatic_restart=True,
                ),
                service_accounts=[sa],
                metadata=compute_v1.Metadata(
                    items=[
                        compute_v1.Items(
                            key="google-logging-enabled",
                            value="true"
                        ),
                        compute_v1.Items(
                            key="install-nvidia-driver",
                            value="true"
                        ),
                        compute_v1.Items(
                            key="ssh-keys",
                            value=f"{current_user}:{pub_key}"  # Replace USER with your desired username
                        ),
                        compute_v1.Items(
                            key="startup-script",
                            value=startup_script
                        )
                    ]
                )
    )

    # Create the instance
    operation = instance_client.insert(
        project=project_id,
        zone=zone,
        instance_resource=instance,
    )
    operation.result()  # Wait for the operation to complete

    return instance_name


def get_machine_type(gpu_memory, gpu_count):
    """
    Determine the appropriate machine type based on GPU memory and count requirements.

    Args:
        gpu_memory (int): Required GPU memory in GB (40 or 80)
        gpu_count (int): Number of GPUs required (1, 2, 4, or 8)

    Returns:
        str: GCP machine type name
    """
    if gpu_memory not in [40, 80]:
        raise ValueError("GPU memory must be either 40GB or 80GB")
    if gpu_count not in [1, 2, 4, 8]:
        raise ValueError("GPU count must be 1, 2, 4, or 8")

    series = "a2-ultragpu" if gpu_memory == 80 else "a2-highgpu"
    return f"{series}-{gpu_count}g"


def main():
    parser = argparse.ArgumentParser(description="Deploy a GPU VM instance on GCP.")
    parser.add_argument(
        "--project-id",
        required=True,
        help="Your GCP project ID"
    )
    parser.add_argument(
        "--zone",
        required=False,
        default="us-west1-b",
        help="The GCP zone to deploy in"
    )
    parser.add_argument(
        "--gpu-memory",
        required=False,
        type=int,
        choices=[40, 80],
        default=40,
        help="GPU memory in GB (40 or 80)"
    )
    parser.add_argument(
        "--gpu-count",
        required=False,
        type=int,
        default=1,
        help="The number of GPUs to use"
    )
    parser.add_argument(
        "--boot-disk-size",
        required=False,
        type=int,
        default=100,
        help="The boot disk size in GB"
    )
    parser.add_argument(
        "--data-disk-size",
        required=False,
        type=int,
        default=100,
        help="The data disk size in GB"
    )
    parser.add_argument(
        "--service-account-path",
        required=False,
        # default=os.path.join(os.environ["HOME"], "key.json"),
        help="Path to service account JSON file"
    )
    args = parser.parse_args()
    machine_type = get_machine_type(args.gpu_memory, args.gpu_count)
    credentials = get_credentials(args.service_account_path)

    # Create environment variables dictionary
    env_vars = {
        "S3_ACCESS_KEY": os.getenv("S3_ACCESS_KEY"),
        "S3_SECRET_KEY": os.getenv("S3_SECRET_KEY")
    }
    # Remove None values
    env_vars = {k: v for k, v in env_vars.items() if v is not None}

    now = datetime.now()
    # Format the date as YYYYMMDD
    formatted_date = now.strftime("%Y%m%d")

    # Create a unique instance name using timestamp
    instance_name = f"{machine_type}-{formatted_date}"

    try:
        print(f"Creating instance {instance_name}...")
        create_instance(
            project_id=args.project_id,
            zone=args.zone,
            instance_name=instance_name,
            machine_type=machine_type,
            gpu_count=args.gpu_count,
            boot_disk_size=args.boot_disk_size,
            data_disk_size=args.data_disk_size,
            env_vars=env_vars,
            credentials=credentials
        )

        time.sleep(20)

        print(f"Instance {instance_name} created successfully!")

        print(f"\nView your instance in the Google Cloud Console:")
        print(
            f"https://console.cloud.google.com/compute/instancesDetail/zones/{args.zone}/instances/{instance_name}?"
            f"project={args.project_id}")

        print("\nTo SSH into your instance, use:")
        print(f"gcloud compute ssh --project {args.project_id} --zone {args.zone} {instance_name}")

        # Interactive command loop
        while True:
            action = input("Enter 't' to terminate the instance or 's' to stop it: ").strip().lower()

            instance_client = compute_v1.InstancesClient()
            if action == 't':
                print("Terminating the instance...")
                operation = instance_client.delete(project=args.project_id, zone=args.zone, instance=instance_name)
                operation.result()
                print("Instance deleted.")
                break

            elif action == 's':
                print("Stopping the instance...")
                operation = instance_client.stop(project=args.project_id, zone=args.zone, instance=instance_name)
                operation.result()
                print("Instance stopped.")
                break

            else:
                print("Invalid input. Please enter 'd' or 's'.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
