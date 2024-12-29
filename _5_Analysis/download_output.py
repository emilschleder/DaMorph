import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from _3_Training.hpc.SSH_client import SSHClient

out_files = [
    'sv_ce_bpe.out',
    'sv_ce_mix.out',
    'sv_ce_mor.out',
    'sv_ce_std.out',
]

ssh_client = SSHClient(
    username=os.environ.get('itu_hpc_username'), 
    password=os.environ.get('itu_hpc_password')
)
ssh_client.connect()

for file in out_files:
    local_file_path = ''
    if file == 'sv_ce_bpe.out':
        local_file_path = 'DA-BPE-CEREBRAS.out.txt'
    elif file == 'sv_ce_mix.out':
        local_file_path = 'DA-MIXED-CEREBRAS.out.txt'
    elif file == 'sv_ce_mor.out':
        local_file_path = 'DA-MORPH-CEREBRAS.out.txt'
    elif file == 'sv_ce_std.out':
        local_file_path = 'STD-BPE-CEREBRAS.out.txt'   
    
    local_file_path_dir = f"_5_Analysis/ScandEval_output/{local_file_path}"
    ssh_client.download_file(file, local_file_path_dir)

ssh_client.close_connection()
