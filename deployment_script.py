import runpod
import time
import argparse

parser = argparse.ArgumentParser(description="Deploy a Python project to RunPod.")
parser.add_argument(
    "--entry-point",
    required=True,
    help="The entry point file to execute in the GitHub repository. e.g anthropic_sft.py"
)
args = parser.parse_args()

# Replace with your RunPod API Key
RUNPOD_API_KEY = "rpa_AWR0U7BFVTL843TA4EQY9VUGN74A6WCMTYW5TPVM24z4zn"

# GitHub repository URL and script settings
GITHUB_REPO_URL = "https://github.com/horus-ai-labs/DistillFlow.git"
ENTRY_POINT = "args.entry_point"

# Initialize RunPod client
runpod.api_key = RUNPOD_API_KEY
print("Endpoints are ", runpod.get_endpoints())
# client = runpod.Client(api_key=RUNPOD_API_KEY)

# Define pod configuration
# pod_config = {
#     "name": "distillflow-pod",
#     "image_name": "python:3.9",  # Base image
#     "pod_type": "CPU_SINGLE",   # Adjust based on your needs
#     "container_disk_in_gb": 20, # Disk size
#     "args": [
#         "/bin/bash", "-c",  # Execute shell commands
#         f"""
#         git clone {GITHUB_REPO_URL} app &&
#         cd app &&
#         curl -sSL https://install.python-poetry.org | python3 - &&
#         export PATH=$HOME/.local/bin:$PATH &&
#         poetry install &&
#         poetry run python {ENTRY_POINT}
#         """
#     ]
# }

# print(runpod.get_gpus())
# Launch the pod
print("Launching pod...")

resp = runpod.create_pod(name="test", image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
                         gpu_type_id="NVIDIA A40", gpu_count=2, start_ssh=True,
                         env={"GITHUB_REPO": GITHUB_REPO_URL,"ENTRY_POINT": ENTRY_POINT},
                         docker_args=(
                             "/bin/bash -c 'git clone ${GITHUB_REPO} app && \
                             cd app && \
                             curl -sSL https://install.python-poetry.org | python3 - && \
                             export PATH=$HOME/.local/bin:$PATH && \
                             poetry install && \
                             poetry run python ${ENTRY_POINT}'"
    ))
# Wait before termination (or remove this to keep the pod running)
pod_id = resp['id']

# while True:
#     describe_pod_resp = runpod.get_pod(pod_id=pod_id)
#     print("describe_pod resp: ", describe_pod_resp)
#
#     pod_status = describe_pod_resp["status"]["phase"]
#     print(f"Current pod status: {pod_status}")
#     if pod_status == "RUNNING":
#         break
#     elif pod_status in ["ERROR", "FAILED"]:
#         print(f"Pod failed to start: {pod_status}")
#         runpod.terminate_pod(pod_id)
#         exit(1)
time.sleep(120)

print("pod launched successfully with id ", pod_id)
input("Press Enter to stop the pod...")
runpod.stop_pod(pod_id)
print("Pod terminated.")

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
