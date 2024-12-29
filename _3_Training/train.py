import itertools
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os, sys, re, json, os, sys, morfessor, deepspeed
import argparse
from datetime import datetime
from huggingface_hub import hf_hub_download, HfApi, HfFolder

# These tokenizer Imports will only work on HPC. 
# Comment out for local testing and comment in the lines below
from DaMorph_raw import MorfessorTokenizer
from DaMorph_mixed import MorfessorBPETokenizer

# Comment in for local testing
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from _2_Tokenizer.DaMorph_tokenizers.DaMorph_raw import MorfessorTokenizer
# from _2_Tokenizer.DaMorph_tokenizers.DaMorph_mixed import MorfessorBPETokenizer

parser = argparse.ArgumentParser(description="Training script for embeddings")
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=1, help='Micro-batch size per GPU')
parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
parser.add_argument('--max_length_To_Give', type=int, default=1024, help='Maximum sequence length')
parser.add_argument('--morf_bpe', action='store_true', help='Use Morfessor BPE tokenizer')
parser.add_argument('--Fintune_Embeddings_Only', action='store_true', help='Finetune embeddings only')
parser.add_argument('--dataset_used', type=str, default="giga_small.txt", help='Path to the dataset file')
parser.add_argument('--tokenizer_name', type=str, default="meelu/morf_tokenizer_mixed.440M_128256", help='Tokenizer name')
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B", help='Model name')
parser.add_argument('--hf_dir', type=str, default="", help='Hugging Face directory')

args = parser.parse_args()

num_epochs = args.num_epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
gradient_accumulation_steps = args.gradient_accumulation_steps
max_length_To_Give = args.max_length_To_Give
morf_bpe = args.morf_bpe
Fintune_Embeddings_Only = args.Fintune_Embeddings_Only
dataset_used = args.dataset_used
tokenizer_name = args.tokenizer_name
model_name = args.model_name
hf_dir = args.hf_dir

print(f"Model: {model_name}")

# TOKENIZER LOAD
if morf_bpe:
    print("Using Morfessor BPE tokenizer")
    tokenizer = MorfessorBPETokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name,
        morfessor_model=True
    )
elif not morf_bpe and 'morf' in tokenizer_name:
    print("Using Morfessor tokenizer")
    tokenizer = MorfessorTokenizer.from_pretrained(
        pretrained_model_name_or_path=tokenizer_name,
        morfessor_model=True
    )
else:
    print("Using AutoTokenizer tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

if Fintune_Embeddings_Only:
    print("Only finetuning embeddings")
else:
    print("Finetuning all layers")

# CHECKPOINT
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_INTERVAL = 2000       
OVERWRITE_CHECKPOINTS = False      

# FIND MAXIMUM SEQUENCE LENGTH
def get_max_position_embeddings(config):
    if hasattr(config, 'n_positions'):
        return config.n_positions
    elif hasattr(config, 'max_position_embeddings'):
        return config.max_position_embeddings
    else:
        raise AttributeError("Cannot find sequence length")

#Concats sentences together to utilize the batch size
def concatenate(dataset_iter, tokenizer, max_length):
    Text_seq = ""
    sample_count = 0

    for example in dataset_iter:
        sentence = example["text"].strip()
        if not sentence:
            continue
    
        concat_seq = f"{Text_seq} {sentence}" if Text_seq else sentence
        tokenized = tokenizer.tokenize(concat_seq, add_special_tokens=True)
        token_ids = tokenizer.convert_tokens_to_ids(tokenized)
        
        #Check if the tokens are over the maximum sequence length
        T_count = len(token_ids)
        if T_count <= max_length:
            Text_seq = concat_seq
            sample_count += 1
        else:
            if Text_seq:
                yield Text_seq, sample_count

            Text_seq = sentence
            sample_count = 1
            new_tokenized = tokenizer.tokenize(Text_seq, add_special_tokens=True)
            new_token_ids = tokenizer.convert_tokens_to_ids(new_tokenized)

            if len(new_token_ids) > max_length:
                truncated_token_ids = new_token_ids[:max_length]
                Text_seq = tokenizer.decode(truncated_token_ids, clean_up_tokenization_spaces=True)
                yield Text_seq, sample_count
                Text_seq = ""
                sample_count = 0

    # Print and yield any remaining sequence
    if Text_seq:
        yield Text_seq, sample_count

#Put all the batches in a list until batch size is reached
def stream_batch(dataset_iter, tokenizer, batch_size, max_length):
    batch = []
    S_counts = []

    for sequence, count in concatenate(dataset_iter, tokenizer, max_length):
        tokenized_sequence = tokenizer(
            sequence,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors=None, 
        )
        batch.append(tokenized_sequence)
        S_counts.append(count)

        #Put together to reach the batch size
        if len(batch) == batch_size:
            batch_padded = tokenizer.pad(
                batch,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )
            yield batch_padded, S_counts
            batch = []
            S_counts = []

    # Yield the last batches
    if batch:
        batch_padded = tokenizer.pad(
            batch,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )
        print(batch_padded)
        yield batch_padded, S_counts

#Regex pattern to upload to HUGGINGFACE
def Regex_files(name):
    return re.sub(r'[^\w\-_.]', '_', name)

def train(rank, world_size, num_epochs, batch_size, tokenizer, max_length_To_Give, unique_id, tokenizer_name):
    
    #NCCL DEBUG env variables
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['NCCL_DEBUG'] = 'INFO' 
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    #dataset init
    dataset = load_dataset(
        'text', 
        data_files=dataset_used,  # Path to your local .txt file
        split='train',
        streaming=True,
    )
    dataset_iter = iter(dataset)
    dataset_shard_iter = itertools.islice(dataset_iter, rank, None, world_size)

    #Setup Deepspeed and GPU's
    torch.cuda.set_device(rank)
    print(f"Rank {rank} batch_size: {batch_size}")
    print(f"Process {rank}: WORLD_SIZE: {world_size}, RANK={rank}")
    print(f"Process {rank} using device {torch.cuda.current_device()}")
    
    ds_config = {
        "zero_optimization": {
            "stage": 0,
            "overlap_comm": True,
        },
        "bf16": {
            "enabled": False  
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 32,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "train_batch_size": batch_size * gradient_accumulation_steps * world_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 6e-4
            }
        },
        "wall_clock_breakdown": False,
        "distributed_training": {
            "backend": "nccl",
            "init_method": "env://"
        }
    }
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    print("model layers before resizing of embeddings")
    print(model)
    model.resize_token_embeddings(len(tokenizer))

    print("model layers after resizing of embeddings")
    print(model)

    #Set the maximum length if possible
    try:
        max_model_length = get_max_position_embeddings(model.config)
        print(f"Max length: {max_model_length}")
    except AttributeError as e:
        max_model_length = 2048 
        print(f"Error - Max length to default: {max_model_length}")

    if max_length_To_Give > max_model_length:
        print(f"Max_length is to high {max_model_length}.")
        max_length_To_Give = max_model_length

    #init embeddings and make the contigous
    input_embeddings = model.get_input_embeddings()
    torch.nn.init.xavier_uniform_(input_embeddings.weight)
    input_embeddings.weight.data = input_embeddings.weight.data.contiguous()
    
    #Freze the param
    for param in model.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()
        if Fintune_Embeddings_Only:
            param.requires_grad = False
        else:
            param.requires_grad = True

    input_embeddings.weight.requires_grad = True

    #puch model to device
    device = torch.device(f"cuda:{rank}")
    model.to(device)

    # Init DeepSpeed 
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=filter(lambda p: p.requires_grad, model.parameters()),
        config=ds_config,
        args=None
    )
    #Sync GPU's and start training
    torch.cuda.synchronize()
    model_engine.train()

    global_step = 0

    try:
        for epoch in range(num_epochs):
            #Keep track of samples
            processed_samples = 0
            total_loss = 0
            batch_count = 0

            data_iter = stream_batch(dataset_shard_iter, tokenizer, batch_size, max_length_To_Give)
            Data_Last_Batch_Stop = False

            while True:
                if not Data_Last_Batch_Stop:
                    try:
                        batch, S_counts = next(data_iter)
                    except StopIteration:
                        #If there is not 
                        Data_Last_Batch_Stop = True

                #If last batch is not equal to gpu's - start creating dummy data (Only on the last batch so the traning dosent fail due to gpu not sync)
                #Not optimal - But works
                exhausted_tensor = torch.tensor([1 if Data_Last_Batch_Stop else 0], device=device)
                torch.distributed.all_reduce(exhausted_tensor, op=torch.distributed.ReduceOp.SUM)
                if exhausted_tensor.item() == world_size:
                    break

                if Data_Last_Batch_Stop:
                    print(f"Rank {rank} is exhausted - Creating dummy data")
                    dummy_input = torch.zeros(1, 1, dtype=torch.long).to(device)
                    outputs = model_engine(dummy_input, labels=dummy_input)
                    loss = outputs.loss * 0
                    model_engine.backward(loss)
                    model_engine.step()
                else:
                    #Normal traning
                    input_ids = batch["input_ids"].cuda()
                    attention_mask = batch["attention_mask"].cuda()
                    labels = input_ids.clone()
                    outputs = model_engine(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                    model_engine.backward(loss)
                    model_engine.step()

                    processed_samples += sum(S_counts)
                    total_loss += loss.item()
                    batch_count += 1
                    global_step += 1

                    # If global step is equal Chekpoint interval - Make a chekpoint
                    if global_step % CHECKPOINT_INTERVAL == 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        checkt_tag = f"batch{global_step}_id{unique_id}_tokenizer{Regex_files(tokenizer_name)}"
                        checkpoint_result = model_engine.save_checkpoint(
                            CHECKPOINT_DIR, 
                            tag=checkt_tag, 
                            client_state={
                                "epoch": epoch,
                                "batch_count": batch_count,
                                "global_step": global_step,
                                "tokenizer_name": tokenizer_name,
                                "unique_id": unique_id
                            }
                        )
                        torch.distributed.barrier()
                    
                    #Print process
                    if batch_count % 100 == 0 and rank == 0:
                        avg_loss = total_loss / batch_count
                        print(f"Epoch {epoch+1}, Batch {batch_count}, Avg Loss: {avg_loss:.4f}, Processed Samples: {processed_samples}")

            torch.distributed.barrier()

    #Save if something fails
    except Exception as e:
        unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rank}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkt_tag = f"error_{timestamp}_id{unique_id}_tokenizer{Regex_files(tokenizer_name)}"

        model_engine.save_checkpoint(
            CHECKPOINT_DIR, 
            tag=checkt_tag, 
            client_state={
                "error": str(e) if rank == 0 else "",  
                "tokenizer_name": tokenizer_name,
                "unique_id": unique_id
            }
        )

        torch.distributed.barrier()

        if rank == 0:
            print(f"Error checkpoint saved")
        raise e 

    #Save model when traning is done and upload to HUGGINGFACE
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_directory = f"./{model_name.split('/')[1].replace('-', '')}_{tokenizer_name.split('/')[1].replace('-', '')}_OnlyEmb_{str(Fintune_Embeddings_Only)}_{timestamp}" 
        model_engine.module.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        hub_model_id = f"{hf_dir}/{model_name.split('/')[1].replace('-', '')}_{tokenizer_name.split('/')[1].replace('-', '')}_OnlyEmb_{str(Fintune_Embeddings_Only)}_{timestamp}" 
        
        api = HfApi()
        token = HfFolder.get_token()
        if token is None:
            print("Cant find token")
        else:
            api.create_repo(repo_id=hub_model_id, exist_ok=True)
            model_engine.module.push_to_hub(hub_model_id)
            if isinstance(tokenizer, MorfessorBPETokenizer):
                morfessor_model_file = hf_hub_download(
                    repo_id=tokenizer_name, 
                    filename="morfessor_model.bin", 
                    token=token
                )
                tokenizer.push_to_hub(
                    hub_model_id, 
                    token=token,
                    morfessor_model_path=morfessor_model_file
                )
            elif isinstance(tokenizer, MorfessorTokenizer):
                morfessor_model_file = hf_hub_download(
                    repo_id=tokenizer_name, 
                    filename="morfessor_model.bin", 
                    token=token
                )

                tokenizer.push_to_hub(
                    hub_model_id, 
                    token=token,
                    morfessor_model_path=morfessor_model_file
                )
            else:
                tokenizer.push_to_hub(
                    hub_model_id, 
                    token=token
                )

    dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"DEVICE COUND: {world_size}")

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = '127.0.0.1' 
    os.environ['MASTER_PORT'] = '29501'      
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_P2P_LEVEL'] = 'NVL' 
    os.environ['NCCL_TIMEOUT'] = '7200'  
    print(f"NCCL_TIMEOUT: {os.environ.get('NCCL_TIMEOUT')}")

    # Make checkpint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Spawn Gpu / Process
    mp.spawn(
        train, 
        args=(world_size, num_epochs, batch_size, tokenizer, max_length_To_Give, unique_id, tokenizer_name), 
        nprocs=world_size, 
        join=True
    )
    
if __name__ == "__main__":
    print("Training - Distributed")
    main()
