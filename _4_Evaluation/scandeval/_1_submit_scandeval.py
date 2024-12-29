import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from _3_Training.hpc.SSH_client import SSHClient
from generate_jobs import generate_jobs
import argparse

parser = argparse.ArgumentParser(description='Submit ScanEval jobs.')
parser.add_argument('--models_file', type=str, required=True, help='Path to the models file.')
parser.add_argument('--hpc_env', type=str, required=True, help='Env to use on hpc')
args = parser.parse_args()

models_file = args.models_file
hpc_env = args.hpc_env

# Initiate SSH client
ssh_client = SSHClient(
    username=os.environ.get('itu_hpc_username'), 
    password=os.environ.get('itu_hpc_password'),
)
ssh_client.connect()

# Uploading models file
ssh_client.upload_file(
    local_path=models_file,
    remote_path=models_file.split('/')[-1]
)

files = generate_jobs(
    hf_token=os.environ.get('huggingface_token'),
    env=hpc_env,
    verbose=True,
    tasks=[
        'linguistic-acceptability'
        , 'summarization'
    ],
    eval_models=models_file
)

for path in files:
    job_name = path.split('/')[-1]

    # Uploading job file
    ssh_client.upload_file(
        local_path=path,
        remote_path=job_name
    )

    # Submitting job
    job_id = ssh_client.submit_job(job_name)

ssh_client.close_connection()