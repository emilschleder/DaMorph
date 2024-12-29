#!/bin/bash

#SBATCH -N 1                        # number of nodes
#SBATCH --job-name=sv_ll_mor       # Job name
#SBATCH --output=sv_ll_mor.out  # Name of output file
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
source activate scandeval_env_3

# Enable device-side assertions
export TORCH_USE_CUDA_DSA=1

# Run script
echo "Scandeval using meelu/DA-MORPH-LLAMA3.2"

# Run scandeval while disabling VLLM to avoid issues with custom tokenizers
USE_VLLM=0 scandeval --model meelu/DA-MORPH-LLAMA3.2 --task linguistic-acceptability --task summarization --language da --no-use-flash-attention --device cuda --force --use-token --token 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 --clear-model-cache --verbose --debug

echo "Done"
