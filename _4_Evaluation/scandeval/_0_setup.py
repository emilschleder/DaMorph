import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from _3_Training.hpc.SSH_client import SSHClient
import argparse

parser = argparse.ArgumentParser(description='Setup script for uploading files to remote server.')
parser.add_argument('--package_path', type=str, required=True, help='Path to the package on the remote server')
parser.add_argument('--local_path', type=str, required=True, help='Path to the local files')

args = parser.parse_args()

package_path = args.package_path
local_path = args.local_path

tokenizers = [
    '_2_Tokenizer/DaMorph_tokenizers/DaMorph_mixed.py',
    '_2_Tokenizer/DaMorph_tokenizers/DaMorph_raw.py',
]

files = {
    'hf_file': {
        'local': os.path.join(local_path, 'hf.py'),
        'remote': os.path.join(package_path, 'model_setups', 'hf.py')
    },
    'generation_file': {
        'local': os.path.join(local_path, 'generation.py'),
        'remote': os.path.join(package_path, 'generation.py')
    },
    'tokenizer_folder': {
        'remote': package_path
    },
    'seqeval_file': {
        'local': os.path.join(local_path, 'sequence_classification.py'),
        'remote': os.path.join(package_path, 'sequence_classification.py')
    },
    'summ_file': {
        'local': os.path.join(local_path, 'text_to_text.py'),
        'remote': os.path.join(package_path, 'text_to_text.py')
    }
}

# Initiate SSH client
ssh_client = SSHClient(
    username=os.environ.get('itu_hpc_username'), 
    password=os.environ.get('itu_hpc_password')
)
ssh_client.connect()

# upload tokenizers
for file in tokenizers:
    ssh_client.upload_file(
        local_path=file,
        remote_path=os.path.join(
            files['tokenizer_folder']['remote'], 
            file.split('/')[-1]
        )
    )

# Upload HF model setup
ssh_client.upload_file(
    local_path=files['hf_file']['local'],
    remote_path=files['hf_file']['remote']
)

# Upload generation file
ssh_client.upload_file(
    local_path=files['generation_file']['local'],
    remote_path=files['generation_file']['remote']
)

# Upload seqeval file
ssh_client.upload_file(
    local_path=files['seqeval_file']['local'],
    remote_path=files['seqeval_file']['remote']
)

# Upload text_to_text file
ssh_client.upload_file(
    local_path=files['summ_file']['local'],
    remote_path=files['summ_file']['remote']
)

# Closing connection
ssh_client.close_connection()