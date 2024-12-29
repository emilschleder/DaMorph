import paramiko
import getpass
import time
import os
from scp import SCPClient

class SSHClient:
    def __init__(
        self, 
        username: str, 
        host: str = 'hpc.itu.dk', 
        password: str = None
    ):
        self.host = host
        self.username = username
        self.password = password
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def connect(self):
        if self.password is None:
            self.password = getpass.getpass(
                f'Enter password for {self.username}@{self.host}: '
            )
        self.ssh.connect(
            hostname=self.host, 
            username=self.username, 
            password=self.password
        )

    def upload_file(
        self, 
        local_path: str, 
        remote_path: str
    ):
        with SCPClient(self.ssh.get_transport()) as scp:
            print(f'Uploading {local_path} to {remote_path}')
            scp.put(
                files=local_path, 
                remote_path=remote_path
            )
            print('Upload complete\n')
    
    def upload_folder(
        self, 
        local_path: str, 
        remote_path: str
    ):
        if not os.path.isdir(local_path):
            raise ValueError(f"{local_path} is not a valid directory")

        with SCPClient(self.ssh.get_transport()) as scp:            
            print(f'Uploading {local_path} to {remote_path}')
            scp.put(
                files=local_path, 
                remote_path=remote_path, 
                recursive=True
            )
            print('Upload complete\n')


    def submit_job(
        self, 
        job_script: str
    ):
        stdin, stdout, stderr = self.ssh.exec_command(f'sbatch {job_script}')
        print(stdout)
        job_id = int(stdout.read().decode().split()[-1])
        print(f'Submitted job with ID {job_id}')
        return job_id

    def wait_for_job_completion(
        self, 
        job_id: int
    ):
        job_finished = False
        while not job_finished:
            time.sleep(30)
            stdin, stdout, stderr = self.ssh.exec_command(f'sacct -j {job_id} --format=State --noheader')
            output = stdout.read().decode().strip().split('\n')
            for line in output:
                if 'COMPLETED' in line:
                    job_finished = True
                    print(f'Job {job_id} has completed')
                    break
                elif 'RUNNING' in line:
                    print(f'Job {job_id} is still running...')
                    break

    def show_output(
        self, 
        job_id: int, 
        show_output: bool = False
    ):
        job_output_filename = f'job.{job_id}.out'
        local_output_path = os.path.join(os.getcwd(), job_output_filename)
    
        with SCPClient(self.ssh.get_transport()) as scp:
            scp.get(job_output_filename, local_output_path)
        print(f'Downloaded .out file to {local_output_path}')
        
        if show_output:
            with open(local_output_path) as file:
                for line in file:
                    print(line)
    
    def download_file(
        self, 
        remote_path: str, 
        local_path: str
    ):
        with SCPClient(self.ssh.get_transport(), socket_timeout=5000) as scp:
            scp.get(
                remote_path=remote_path, 
                local_path=local_path
            )
        print(f'Downloaded file {remote_path} to {local_path}')
    
    def close_connection(self):
        self.ssh.close()