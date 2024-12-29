import math
import morfessor
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

parser = argparse.ArgumentParser(description='Train Morfessor models with different dataset sizes.')
parser.add_argument('--data_path', type=str, required=True, help='Base path to the dataset files')
parser.add_argument('--output_model_path', type=str, required=True, help='Path to save the trained models')
parser.add_argument('--corpus', type=str, required=True, help='Path to the corpus file')

args = parser.parse_args()

data_path = args.data_path
output_model_path = args.output_model_path
corpus_path = args.corpus

def log_func(x):
    return int(round(math.log(x + 1, 2)))

def train_and_save_model(size, data_path, output_model_path, corpus):
    io = morfessor.MorfessorIO()
    trainer = morfessor.BaselineModel()

    if size != 0:
        annotated_training_data_path = os.path.join(data_path, f"{size}.txt")
        annotated_data = io.read_annotations_file(annotated_training_data_path)
        trainer.set_annotations(annotated_data)
   
    model_output_path = os.path.join(output_model_path, f"Model{size}_{data_path.split("/")[-1]}.bin")
    
    print(f"Training for dataset: {size}.txt...")
    
    trainer.load_data(corpus, count_modifier=log_func)
    trainer.train_batch()
    
    io.write_binary_model_file(model_output_path, trainer)
    print(f"Model trained and saved as: {model_output_path}")

if __name__ == '__main__':
    dataset_sizes = [0, 50, 100, 150, 200, 250, 300, 350, 400]
    io = morfessor.MorfessorIO()
    corpus = list(io.read_corpus_file(corpus_path))

    # Parallel training
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(train_and_save_model, size, data_path, output_model_path, corpus): size for size in dataset_sizes}
        for future in as_completed(futures):
            size = futures[future]
            try:
                future.result()
                print(f"Completed for dataset: {size}.txt")
            except Exception as e:
                print(f"Failed for dataset: {size}.txt with error: {e}")
    
    print("\nAll models have been trained and saved.")