import re, os, json, morfessor, time, logging
from collections import defaultdict
from datasets import load_dataset
from DaMorph_tokenizers.DaMorph_raw import MorfessorTokenizer
from DaMorph_tokenizers.DaMorph_mixed import MorfessorBPETokenizer
import argparse

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Tokenizer pipeline arguments")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
parser.add_argument("--morf_vocab_size", type=int, required=True, help="Vocabulary size for Morfessor")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
parser.add_argument("--morf_bpe", action='store_true', help="Use Morfessor BPE tokenizer")
parser.add_argument("--bpe_tokenizer", type=str, default=None, help="BPE tokenizer path")
parser.add_argument("--bpe_tokenizer_size", type=int, default=0, help="BPE tokenizer size")
parser.add_argument("--morfessor_model_path", type=str, default=None, help="Path to Morfessor model")
parser.add_argument("--morf_table_file", type=str, default=None, help="Path to morf table file")
parser.add_argument("--chunks_file", type=str, default=None, help="Path to chunks file")
parser.add_argument("--hf_dir", type=str, default=None, help="Huggingface directory")

args = parser.parse_args()

dataset_name = args.dataset_name
morf_vocab_size = args.morf_vocab_size
output_dir = args.output_dir
morf_bpe = args.morf_bpe
bpe_tokenizer = args.bpe_tokenizer
bpe_tokenizer_size = args.bpe_tokenizer_size
morfessor_model_path = args.morfessor_model_path
chunks_file = args.chunks_file
morf_table_file = args.morf_table_file
hf_dir = args.hf_dir

working_dir = f'{output_dir}/morf_{"bpe_" if morf_bpe else ""}tokenizer_{time.strftime("%Y%m%d_%H%M%S")}'
vocab_size = morf_vocab_size + bpe_tokenizer_size
hf_tokenizer_name = f"morf_{"bpe_" if morf_bpe else ""}tokenizer_{dataset_name.split("/")[-1]}_{vocab_size}"
hf_token = os.environ.get("huggingface_token")


# Step 1: Load the dataset
def _1_extract_chunks_and_count(
    dataset_name, 
    dataset_split='train'
):
    # Load the dataset 
    dataset = load_dataset(
        dataset_name, 
        split=dataset_split, 
        streaming=True
    )
    word_counts = defaultdict(int)

    logging.info("Loaded dataset")
    
    # Function to extract chunks using regex 
    def extract_words(text):
        words = re.split(r'(?=\s)', text) 
        for word in words:
            word_counts[word] += 1  

    for i, sample in enumerate(dataset):
        if i%100000 == 0:
            print(f"processed {i} samples")
        extract_words(sample["text"])

    output_dir_ = os.path.join(output_dir, "chunk_counts.json")
    
    # Save the word counts as a JSON file
    with open(output_dir_, "w") as f:
        json.dump(word_counts, f, indent=4)
        
    logging.info(f"Saved {len(word_counts)} unique chunks with their counts to {output_dir_}")
    return output_dir_

def _2_sort_chunks(
    working_dir, 
    input_file, 
    output_file='chunks.json'
):   

    # Load the original JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logging.info("Original JSON file loaded successfully.\nInitiating sorting...")
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    
    logging.info("Sorting completed successfully.\nInitiating saving of new JSON file...")

    save_path = os.path.join(output_dir, output_file)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=4)
    
    logging.info("New JSON file with top chunks created successfully.")
    return save_path

def _3_build_morf_table(
    input_filename, 
    morfessor_model_path
):
    word_data = {}
    count = 0

    io = morfessor.MorfessorIO()
    model = io.read_binary_model_file(morfessor_model_path)
    
    logging.info("Morfessor model loaded successfully.")
    
    with open(input_filename, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    logging.info("JSON file loaded successfully.")
    logging.info("Initiating morpheme segmentation...")

    for word, count in json_data.items():
        morphemes = model.viterbi_segment(word)[0]
        word_data[word] = {
            "count": count,  
            "segment": morphemes
        }

    sorted_word_data = {k: v for k, v in sorted(word_data.items(), key=lambda item: item[1]['count'], reverse=True)}
    output_filename = f"morf_table_{count}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_dir_ = os.path.join(output_dir, output_filename)
    
    # Write the output to a JSON file
    with open(output_dir_, 'w', encoding='utf-8') as outfile:
        json.dump(sorted_word_data, outfile, ensure_ascii=False, indent=4)

    logging.info(f'Morphemes have been processed and saved to {output_filename}.')
    return output_filename

def _4_build_vocab(
    morf_table_path,
    save_directory,
    tokenizer,
    tokenizer_name,
    morfessor_path,
    hf_token,
    hf_dir,
    morph_vocab_size=1,
    bpe_tokenizer_path=None,
    bpe=False
):
    if not bpe:
        logging.info("Building Morf vocabulary...")
        tokenizer.build_vocab(
            morph_table_path=morf_table_path, 
            morph_vocab_size=morph_vocab_size
        )
        tokenizer.save_vocabulary(
            save_directory=save_directory,
            filename_prefix=f"morf_vocab_{morph_vocab_size}_{time.strftime('%Y%m%d_%H%M%S')}",
            push_to_hub=True,
            hf_repo_name=f"{hf_dir}/{tokenizer_name}",
            morfessor_model_path=morfessor_path,
            token=hf_token
        )
    else:
        logging.info("Building Morf_BPE vocabulary...")
        tokenizer.build_vocab(
            morph_table_path=morf_table_path, 
            morph_vocab_size=morph_vocab_size,
            bpe_tokenizer_path=bpe_tokenizer_path
        )
        tokenizer.save_vocabulary(
            save_directory=save_directory,
            filename_prefix=f"bpe_vocab_{morph_vocab_size}_{time.strftime('%Y%m%d_%H%M%S')}",
            push_to_hub=True,
            hf_repo_name=f"{hf_dir}/{tokenizer_name}",
            token=hf_token
        )
    
if __name__ == '__main__':
    
    logging.info(f"Starting the pipeline for {'morf_bpe' if morf_bpe else 'morf'} tokenizer...")
    
    if morf_bpe:
        tokenizer = MorfessorBPETokenizer()
    else:
        tokenizer = MorfessorTokenizer()
        
    # Create working directory if not exists
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    
    if not chunks_file:    
        # 1. Extract chunks and count
        chunky_file = _1_extract_chunks_and_count(
            dataset_name = dataset_name
        )
        
        # 2. Sort chunks
        sorted_chunks_file = _2_sort_chunks(
            working_dir=working_dir, 
            input_file=chunky_file
        )
    else:
        sorted_chunks_file=chunks_file
        
    if not morf_table_file:
        # 3. Build morpheme table
        morf_table_path = _3_build_morf_table(
            input_filename = sorted_chunks_file,
            morfessor_model_path = morfessor_model_path
        )
    else: 
        morf_table_path=morf_table_file
        
    # 4. Build vocabulary
    _4_build_vocab(
        morf_table_path=morf_table_path,
        save_directory=working_dir,
        morph_vocab_size=morf_vocab_size,
        tokenizer=tokenizer,
        bpe_tokenizer_path=bpe_tokenizer if morf_bpe else None,
        tokenizer_name=hf_tokenizer_name,
        morfessor_path=morfessor_model_path,
        hf_token=hf_token,
        hf_dir=hf_dir,
        bpe=morf_bpe
    )