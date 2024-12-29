from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast
import argparse, os

parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
parser.add_argument('--save_path', type=str, required=True, help='Path to save the trained tokenizer')
parser.add_argument('--vocab_size', type=int, required=True, help='Vocabulary size for the tokenizer')
parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset files')

args = parser.parse_args()
save_path = args.save_path
vocab_size = args.vocab_size
data_path = args.data_path

def build_file_list(data_folder):
    training_files = []
    for folder in os.listdir(data_folder):
        current_folder = os.path.join(data_folder, folder)
        
        if not any(term in current_folder.split('/') for term in ['reddit.da']):
            if os.path.isdir(current_folder):  # Directory
                    training_files.extend([os.path.join(current_folder, f) for f in os.listdir(current_folder) if f.endswith('.txt') or f.endswith('.danish')])
                    
            elif os.path.isfile(current_folder):  # File
                if current_folder.endswith('.txt') or current_folder.endswith('.danish'):
                    training_files.append(current_folder)
    
    print(f"Found {len(training_files)} training files")
    
    for f in training_files:
        print(f)
    return training_files

def train_bpe_tokenizer(save_path):
    print("initializing tokenizer with Byte-Pair Encoding (BPE)")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=True
    )

    # Set up the trainer with special tokens and vocabulary size
    trainer = trainers.BpeTrainer(
        special_tokens=[
            "<|begin_of_text|>", 
            "<|end_of_text|>", 
        ],
        vocab_size=vocab_size,
    )

    print("Building file list")
    files = build_file_list(
        data_folder=data_path
    )

    print("Training tokenizer")
    tokenizer.train(
        files=files, 
        trainer=trainer
    )

    tokenizer.decoder = decoders.ByteLevel(
        add_prefix_space=True
    )

    tokenizer.post_processor = processors.ByteLevel(
        trim_offsets=True
    )

    tokenizer.save(
        path=save_path
    )

    print("Tokenizer training completed and saved as tokenizer")

    backend_tok = Tokenizer.from_file(
        path=save_path
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend_tok
    )

    # Add special tokens
    tok.add_special_tokens({
        'unk_token': '[UNK]',
        'pad_token': '[PAD]',
        'cls_token': '[CLS]',
        'sep_token': '[SEP]',
        'mask_token': '[MASK]',
    })

    # Save the tokenizer in Hugging Face format
    tokenizer_name = f'tokenizer_bpe_{vocab_size}.json'
    tok.save_pretrained(
        save_directory=os.path.join(save_path, tokenizer_name)
    )

    print(f"Tokenizer vocab size: {tok.vocab_size}")

def main():
    train_bpe_tokenizer(save_path)

if __name__ == '__main__':
    main()