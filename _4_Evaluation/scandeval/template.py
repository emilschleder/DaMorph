template = """#!/bin/bash

#SBATCH -N 1                        # number of nodes
#SBATCH --job-name={job_name}       # Job name
#SBATCH --output={job_name}.out  # Name of output file
#SBATCH --cpus-per-task=1           # Schedule cpus
#SBATCH --mem=35G                   # memory per node
#SBATCH --time=1-00:00:00           # Max Run time (hh:mm:ss)
#SBATCH --partition=scavenge        # Run on either the Red or Brown queue
#SBATCH --gres=gpu:1              # If you are using CUDA dependent packages
#SBATCH --mail-type=BEGIN,FAIL,END  # Send an email 

# Current node
echo "Running on: $(hostname)"

# Loading Anaconda
echo "Loading Anaconda"
module load Anaconda3

# Sourcing .bashrc
echo "Sourcing .bashrc"
source /home/easc/.bashrc

# Activate environment
echo "Activating virtual environment"
source activate {env}

# Enable device-side assertions
export TORCH_USE_CUDA_DSA=1

# Run script
echo "Scandeval using {model_name}"

# Run scandeval while disabling VLLM to avoid issues with custom tokenizers
USE_VLLM=0 scandeval --model {model_name} {tasks} --language da --no-use-flash-attention --device cuda --force --use-token --token {hf_token} --clear-model-cache {verbose}

echo "Done"
"""