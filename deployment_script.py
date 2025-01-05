import runpod
import time
import argparse
import os

parser = argparse.ArgumentParser(description="Deploy a Python project to RunPod.")
# parser.add_argument(
#     "--entry-point",
#     required=True,
#     help="The entry point file to execute in the GitHub repository. e.g anthropic_sft.py"
# )
parser.add_argument(
    "--gpu-type",
    required=True,
    default="NVIDIA A40",  # Example default value
    help="The GPU type to use. Defaults to 'NVIDIA A40' if not specified."
)
parser.add_argument(
    "--gpu-count",
    required=False,
    default=2,
    help="The GPU type to use. Defaults to 2 if not specified."
)
parser.add_argument(
    "--volume-in-gb",
    required=False,
    default=50,
    help="The container volume in gb"
)
parser.add_argument(
    "--container-disk-in-gb",
    required=False,
    default=50,
    help="The container disk in gb"
)

args = parser.parse_args()

# Replace with your RunPod API Key
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

# GitHub repository URL and script settings
GITHUB_REPO_URL = "https://github.com/horus-ai-labs/DistillFlow.git"

# Initialize RunPod client
runpod.api_key = RUNPOD_API_KEY
print("Endpoints are ", runpod.get_endpoints())
# print(runpod.get_gpus())
# Launch the pod
print("Launching pod...")

resp = runpod.create_pod(name="pod for llama test", image_name="horuslabs/distillflow:runpod-linux-v3",
                         gpu_type_id=args.gpu_type, gpu_count=args.gpu_count, start_ssh=True, volume_in_gb=args.volume_in_gb,container_disk_in_gb=args.container_disk_in_gb,
                         env={"GITHUB_REPO": GITHUB_REPO_URL, "S3_ACCESS_KEY": S3_ACCESS_KEY, "S3_SECRET_KEY": S3_SECRET_KEY})
pod_id = resp['id']

time.sleep(60)

print("pod launched successfully with id ", pod_id)
while True:
    action = input("Enter 't' to terminate the pod or 's' to stop the pod: ").strip().lower()
    if action == 't':
        print("Terminating the pod...")
        runpod.terminate_pod(pod_id)
        print("Pod terminated.")
        break

    elif action == 's':
        print("Stopping the pod...")
        runpod.stop_pod(pod_id)
        print("Pod stopped.")
        break

    else:
        print("Invalid input. Please enter 't' or 's'.")


# pod_id = runpod.create_pod(name="test", image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04", gpu_type_id="NVIDIA A40", gpu_count=2,
#                            docker_args=(
#         "/bin/bash -c "
#         f"\"git clone {GITHUB_REPO_URL} app && "
#         "cd app && "
#         "curl -sSL https://install.python-poetry.org | python3 - && "
#         "export PATH=$HOME/.local/bin:$PATH && "
#         "poetry install && "
#         f"poetry run python {ENTRY_POINT}\""
#     ))

# Wait for the pod to become ready
# print(f"Pod launched with ID: {pod_id}")
# while True:
#     status = client.get_pod_status(pod_id)["status"]
#     print(f"Current status: {status}")
#     if status == "RUNNING":
#         break
#     time.sleep(5)
#
#
# # Retrieve SSH connection details (optional)
# pod_details = client.get_pod_details(pod_id)
# ssh_command = pod_details.get("ssh_command", "SSH command not available.")
# print(f"SSH command: {ssh_command}")


#  \
#                              poetry run python ${ENTRY_POINT}