```
 _______       ___      .___  ___.   ______   .______      .______    __    __  
|       \     /   \     |   \/   |  /  __  \  |   _  \     |   _  \  |  |  |  | 
|  .--.  |   /  ^  \    |  \  /  | |  |  |  | |  |_)  |    |  |_)  | |  |__|  | 
|  |  |  |  /  /_\  \   |  |\/|  | |  |  |  | |      /     |   ___/  |   __   | 
|  '--'  | /  _____  \  |  |  |  | |  `--'  | |  |\  \----.|  |      |  |  |  | 
|_______/ /__/     \__\ |__|  |__|  \______/  | _| `._____|| _|      |__|  |__| 
                                                                               
```                                                                                                            

Authors: Emil Schledermann, Mikkel Wildner Kildeberg & Nicolaj Larsen

# DaMorph
Danish Morphological Tokenizer

The Following README contains a description on how to re-run the experiments.

## Prerequisites
In order to seamlessly run all experiments, including those on ITU HPC, we strongly suggest that you add these variables to your environment:

    "huggingface_token"

    "itu_hpc_username"

    "itu_hpc_password"

## Running the experiments

The experiments can easily be ran by copying the relative path of the script and running it in the terminal if you are in the projects root directory.

Once you navigate to the project root directory (DaMorph), you need to install the requirements to run experiments. From here you can run all scripts and experiments after installation.

In order to create an environment and install the requirements, you can follow this example:

### Example
This example shows how to easily get started with running our experiments.
It assumes that you have python installed and are using Anaconda. 


```  
# Clone repo and 
git clone https://github.com/emilschleder/DaMorph
cd DaMorph

# Setup environment 
conda create -n damorph_env python=3.12 
conda activate damorph_env 
conda install pip 
pip install -r requirements.txt

# Build morfessor predictions graph
_1_MorfessorModel/_2_Evaluate/_2_generate_plot.sh
``` 

### Experiments scripts

__IMPORTANT NOTE__

    Some scripts require substantial computing power, and needs to run on the ITU hpc. These scripts have been marked with (HPC), and requires a special conda environment, which can be found in _3_Training/hpc/hpc_env.yaml for the training and in _4_Evaluation/scandeval/hpc_scandeval_env.yaml for running scandeval evaluation.

The following shell scripts are available to run:

    _1_MorfessorModel:
        _1_MorfessorModel/_0_COR/_0_prepare_COR.sh
        _1_MorfessorModel/_1_Training/_1_train_models.sh
        _1_MorfessorModel/_2_Evaluate/_1_generate_predictions.sh
        _1_MorfessorModel/_2_Evaluate/_2_generate_plot.sh
        _1_MorfessorModel/run_all_morfessor.sh

    _2_Tokenizer:
        _2_Tokenizer/_1_build_bpe.sh (Requires a folder with txt files to train on)
        _2_Tokenizer/_2_build_DaMorph.sh

    _3_Training:
        _3_Training/_1_train_cerebras.sh (HPC - hpc_env.yaml)
        _3_Training/_1_train_llama.sh (HPC - hpc_env.yaml)

    _4_Evaluation:
        _4_Evaluation/_1_tokenizer_eval.sh
        _4_Evaluation/_2_bytes_eval.sh (HPC - hpc_env.yaml)
        _4_Evaluation/_3_scandeval_eval.sh (HPC - hpc_scandeval_env.yaml)
        _4_Evaluation/_4_human_eval.sh (HPC - hpc_env.yaml)

    _5_Analysis:
        _5_Analysis/_1_extract_results.sh (Requires access to ITU HPC to download .out-files)


## Description of the scripts:

    _1_MorfessorModel:
        _0_COR:
            _0_prepare_COR.sh: Prepares and cleans the COR dataset (from the danish central word register) for training
        _1_Training:
            _1_train_models.sh: Trains the Morfessor models
        _2_Evaluate:
            _1_generate_predictions.sh: Generates predictions for the Morfessor models
            _2_generate_plot.sh: Generates a plot of the Morfessor models
        
        run_all_morfessor.sh: Runs all the Morfessor scripts
        
    _2_Tokenizer:
        _1_build_bpe.sh: Builds the DaMorph Mixed tokenizer
        _2_build_DaMorph.sh: Builds the DaMorph Raw tokenizer
        
    _3_Training:
        _1_train_cerebras.sh: Trains the Cerebras model
        _1_train_llama.sh: Trains the LLAMA model

    _4_Evaluation:
        _1_tokenizer_eval.sh: Evaluates the tokenizers splitting based on evalation data
        _2_bytes_eval.sh: Evaluates the tokenizers on BPC and BPT
        _3_scandeval_eval.sh: Evaluates on ScanEval
        _4_human_eval.sh: Generates prompt for human evaluation
    
    run_all_evals.sh: Runs all the evaluation scripts

    _5_Analysis:
        _1_extract_results.sh: Extracts the results from the ScandEval evaluation


## Models and tokenizers are available at the two collections:
The models are available on Huggingface:

__Model Collection:__ https://huggingface.co/collections/meelu/damorph-676fc4681265035eb20a0ad1

__Tokenizer Collection:__ https://huggingface.co/collections/meelu/damorphtokenizers-67249e829810ecd725d218c5

All of the models can be used locally by cloning this repository and using the following approach within the project folder:

```python
import torch
from _2_Tokenizer.DaMorph_tokenizers.DaMorph_mixed import MorfessorBPETokenizer
from _2_Tokenizer.DaMorph_tokenizers.DaMorph_raw import MorfessorTokenizer
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path="meelu/DA-MIXED-CEREBRAS"
)
tokenizer = MorfessorBPETokenizer.from_pretrained(
    "meelu/DA-MIXED-CEREBRAS", 
    morfessor_model=True
)

# Comment in if you wish to use the raw tokenizer
# model = AutoModelForCausalLM.from_pretrained(
#   pretrained_model_name_or_path="meelu/DA-MORPH-CEREBRAS"
# )
# tokenizer = MorfessorTokenizer.from_pretrained(
#     "meelu/DA-MORPH-CEREBRAS", 
#     morfessor_model=True
# )

text = "Mikkel er en dreng der"

device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(text, return_tensors="pt").to(device)

output = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_length=80,
    do_sample=True,
    no_repeat_ngram_size=2
)

print(tokenizer.decode(token_ids=output[0], skip_special_tokens=True))
```

## Contact
If you encounder any issues, please reach out to us at:

easc@itu.dk 

mwki@itu.dk

nicla@itu.dk