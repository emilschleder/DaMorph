import torch
from torch.utils.data import DataLoader
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from _2_Tokenizer.DaMorph_tokenizers.DaMorph_raw import MorfessorTokenizer
from _2_Tokenizer.DaMorph_tokenizers.DaMorph_mixed import MorfessorBPETokenizer
import logging
from models import models
import argparse

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Evaluate causal language models.")
parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file.')
parser.add_argument('--output_dir', type=str, required=True, help='Path to the output results file.')
args = parser.parse_args()

data_path = args.data_path
output_dir = args.output_dir

logging.info(f"Reading data from {data_path}")

def calc_BPC_BPT(model_info, texts, batch_size=32):
    total_bytes = 0.0
    total_tokens = 0
    Total_characters = 0.0 
    Loss_total = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if 'MORPH' in model_info['tokenizer']:
        logging.info(f"Using MorfessorTokenizer for {model_info['name']}")
        tokenizer = MorfessorTokenizer.from_pretrained(
            model_info['tokenizer'], 
            morfessor_model=True
        )
    elif 'MIXED' in model_info['tokenizer']:
        logging.info(f"Using MorfessorBPETokenizer for {model_info['name']}")
        tokenizer = MorfessorBPETokenizer.from_pretrained(
            model_info['tokenizer'],
            morfessor_model=True
        )
    else:
        logging.info(f"Using AutoTokenizer for {model_info['name']}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_info['tokenizer']
        )
    model = AutoModelForCausalLM.from_pretrained(model_info['name']).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = model.config.max_position_embeddings
    data_loader = DataLoader(texts, batch_size=batch_size)

    logging.info(f"Calculating metrics for {model_info['name']}")
    for batch in data_loader:
        tokenized_input = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
        input_ids = tokenized_input['input_ids'].to(device)
        attention_mask = tokenized_input['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=input_ids
            )
             
            # Shifting for the next token prediction
            guesses = outputs.logits[:, :-1, :].contiguous()
            answers = input_ids[:, 1:].contiguous()
            mask = attention_mask[:, 1:].contiguous()

            # Calculate the loss
            loss_func = torch.nn.CrossEntropyLoss(
                reduction='none'
            )
            loss = loss_func(
                guesses.view(-1, guesses.size(-1)),
                answers.view(-1)
            ).view(answers.size())

            m_loss = loss * mask
            batch_tokens = mask.sum().item()
            bat_loss = m_loss.sum().item()
            
            Loss_total += bat_loss
            total_tokens += batch_tokens

            # Compute Total_characters
            valid_answers = answers[mask == 1]
            decoded_text = tokenizer.decode(
                valid_answers.tolist(), 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            batch_characters = len(decoded_text.replace(' ', ''))
            Total_characters += batch_characters

    if total_tokens == 0 or Total_characters == 0:
        return {
            'model': model_info['name'], 
            'bytes_token': None, 
            'bytes_char': None
        }

    # Calculate bytes per Token and bytes per Character
    total_bytes = Loss_total / math.log(2)
    bytes_token = total_bytes / total_tokens
    bytes_char = total_bytes / Total_characters

    result = {
        'model': model_info['name'],
        'bytes_token': round(bytes_token, 4),
        'bytes_char': round(bytes_char, 4)
    }
    logging.info(result)
    return result

with open(data_path, 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f if line.strip()]

# Evaluate models
results = []
for model in models:
    metrics_result = calc_BPC_BPT(
        model_info=model,
        texts=texts,
        batch_size=64
    )
    results.append(metrics_result)

logging.info("Results:")
logging.info(json.dumps(results, indent=2))

with open(output_dir, 'w', encoding='utf-8') as result_file:
    json.dump(results, result_file, indent=2)
    print(f"Results saved to {output_dir}")