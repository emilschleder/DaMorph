import os
import morfessor

def segment_words_with_models(models_folder, words_file_path, output_folder):
    
    os.makedirs(output_folder, exist_ok=True)
    io = morfessor.MorfessorIO()
    model_files = [f for f in os.listdir(models_folder)]

    with open(words_file_path, 'r', encoding='utf-8') as words_file:
        words = [line.split('\t')[0] for line in words_file]

    for model_file in model_files:
        model_path = os.path.join(models_folder, model_file)
        model = io.read_binary_model_file(model_path)
        
        output_file_path = os.path.join(output_folder, f"segmented_{model_file.replace('.bin', '.txt')}")
        
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            print(f"Segmenting words with model {model_file}...")
            for word in words:
                segments = model.viterbi_segment(word)[0]
                output_file.write(" ".join(segments) + "\n")
            print(f"Segments saved to {output_file_path}")

def main():
    segment_words_with_models(
        models_folder = '_1_MorfessorModel/Models',
        words_file_path = '_1_MorfessorModel/Data/evaluation/71_Words.txt',
        output_folder = '_1_MorfessorModel/Preds'
    )

if __name__ == '__main__':
    main()
