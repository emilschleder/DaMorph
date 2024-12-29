from transformers import AutoTokenizer
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from _2_Tokenizer.DaMorph_tokenizers.DaMorph_raw import MorfessorTokenizer
from _2_Tokenizer.DaMorph_tokenizers.DaMorph_mixed import MorfessorBPETokenizer
from models import models as tokenizer_list
import argparse

parser = argparse.ArgumentParser(description="Evaluate tokenizers")
parser.add_argument('--output_dir', type=str, default='._4_Evaluation/results/results_tokenizer.json', help='Path to save evaluation scores')
parser.add_argument('--eval_file_path', type=str, default='./_1_MorfessorModel/Data/evaluation/424_Words.txt', help='Path to the evaluation file')
args = parser.parse_args()

output_dir = args.output_dir
eval_file_path = args.eval_file_path
        
def get_category(line_number, category_ranges):
    for start, end, category in category_ranges:
        if start <= line_number <= end:
            return category
    return 'Unknown'

def read_target_data(file_path, category_ranges):
    target_data = []
    categories_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            tok = line.strip().split()
            if len(tok) < 2:
                print(f"Skipping wierd/wrong line {line_number}: {line.strip()}")
                continue 
            word = tok[0]
            morphemes = tok[1:]
            target_data.append(morphemes)
            category = get_category(line_number, category_ranges)
            categories_list.append(category if category else 'Unknown')
    
    print(f"Total words read: {len(target_data)}")
    print(f"Unique categories found: {sorted(set(categories_list))}")
    return target_data, categories_list

def compute_target_spans(morphemes):
    spans_set = set()
    start = 0
    for morpheme in morphemes:
        morpheme = morpheme.replace('-', '')
        stop = start + len(morpheme)
        spans_set.add((start, stop))
        start = stop
    return spans_set

def compute_predicted_spans(word_str, tokens):
    spans_set = set()
    current = 0
    for token in tokens:
        if token == '-':
            current += 1  # Length of hyphen
            continue
        # Find the token in the word_str starting from the current position
        start = word_str.find(token, current)
        if start == -1:
            continue
        stop = start + len(token)
        spans_set.add((start, stop))
        current = stop
    return spans_set

def main():
    category_ranges = [
        (1, 145, 'Root Morphemes'),
        (146, 247, 'Compounds'),
        (248, 311, 'Compounds with Linking'),
        (312, 363, 'Prefixes'),
        (364, 397, 'Suffixes'),
        (398, 424, 'Inflection'),
    ]
    category_size = {
        'Root Morphemes': 145,
        'Compounds': 102,
        'Compounds with Linking': 64,
        'Prefixes': 52,
        'Suffixes': 34,
        'Inflection': 27
    }
    
    target_data, categories_list = read_target_data(
        file_path=eval_file_path, 
        category_ranges=category_ranges
    )

    if not target_data:
        print("No target data found.")
        return

    if not categories_list:
        print("No categories found.")
        return

    unique_categories = sorted(set(categories_list))
    print(f"Categories to evaluate: {unique_categories}")
    if not unique_categories:
        print("No categories to evaluate.")
        return
    
    scores_json = {}
    
    # Evaluate each tokenizer
    for tokenizer_entry in tokenizer_list:
        print(f"Tokenizer entry: {tokenizer_entry}")
        try:
            if tokenizer_entry['tokenizer_type'] == 'MorfessorTokenizer':
                tokenizer = MorfessorTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_entry['tokenizer'],
                    morfessor_model=True
                )
            elif tokenizer_entry['tokenizer_type'] == 'MorfessorBPETokenizer':
                tokenizer = MorfessorBPETokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_entry['tokenizer'],
                    morfessor_model=True
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_entry['tokenizer'])

            print(f"\nEvaluating tokenizer: {tokenizer_entry['tokenizer']}")
        except Exception as e:
            print(f"Failed to load tokenizer {tokenizer_entry['tokenizer']}: {e}")
            continue

        # Initialize metrics
        total_target = {category: 0 for category in unique_categories}
        total_pred = {category: 0 for category in unique_categories}
        total_overlap = {category: 0 for category in unique_categories}
        total_accuracy = {category: [] for category in unique_categories}
        correct_predictions = {category: 0 for category in unique_categories}

        # Process each word
        for word_idx, (word_morphemes, category) in enumerate(zip(target_data, categories_list), start=1):
            # Tokenize the word after joining morphemes
            word_str = ''.join(word_morphemes)
            splits = tokenizer.tokenize(word_str, add_special_tokens=False)
            
            if 'MIXED' in tokenizer_entry['tokenizer']:
                splits = [x.replace('§BPE§', '') for x in splits]
                splits = [x.replace('Ã¸', 'ø').replace('Ã¦', 'æ').replace('Ãå', 'å') for x in splits]
                
            elif 'BPE' in tokenizer_entry['tokenizer']:
                splits = [x.replace('Ã¸', 'ø').replace('Ã¦', 'æ').replace('Ãå', 'å') for x in splits]
                splits = [x.replace('Ġ', '') for x in splits]
                if splits and splits[0] in ["<|begin_of_text|>", "<s>", "<|startoftext|>"]:
                    splits = splits[1:]

            predicted_morphemes = splits
            
            target_spans = compute_target_spans(
                morphemes=word_morphemes
            )
            pred_spans = compute_predicted_spans(
                word_str=word_str, 
                tokens=predicted_morphemes
            )
            overlap = len(target_spans.intersection(pred_spans))

            # Update counts
            total_target[category] += len(target_spans)
            total_pred[category] += len(pred_spans)
            total_overlap[category] += overlap

            # Check exact match for accuracy
            is_correct = (predicted_morphemes == word_morphemes)
            total_accuracy[category].append(is_correct)

            if is_correct:
                correct_predictions[category] += 1

        # Calculate scores
        scores = {category: {} for category in unique_categories} # placeholder for scores
        for category in unique_categories:
            
            # Precision
            if total_pred[category] == 0:
                precision = 0.0
            else:
                precision = total_overlap[category] / total_pred[category]
            
            # Recall
            if total_target[category] == 0:
                recall = 0.0
            else:
                recall = total_overlap[category] / total_target[category]
            
            # F1 Score
            if (precision + recall) > 0:
                f1 = 2 * ((precision * recall) / (precision + recall))
            else:
                f1 = 0.0

            # Accuracy
            if total_target[category] > 0:
                accuracy = (correct_predictions[category] / category_size[category]) * 100
            else:
                accuracy = 0.0

            scores[category] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
            }
              
        # Calculate average scores across all categories
        if len(scores) == 0:
            print("No scores to average. Skipping average calculation.")
            avg_recall = 0.0
            avg_precision = 0.0
            avg_f1 = 0.0
            avg_accuracy = 0.0
        else:
            print("len scores", len(scores))
            avg_recall = sum(scores[category]['recall'] for category in scores) / len(scores)
            avg_precision = sum(scores[category]['precision'] for category in scores) / len(scores)
            avg_f1 = sum(scores[category]['f1'] for category in scores) / len(scores)
            avg_accuracy = sum(scores[category]['accuracy'] for category in scores) / len(scores)

        scores['avg.'] = {
            'f1': avg_f1,
            'precision': avg_precision,
            'recall': avg_recall,
            'accuracy': avg_accuracy,
        }

        print("\nToken-based Precision, Recall, F1 Score, and Accuracy:")
        print(f"Precision: {scores['avg.']['precision']:.4f}, Recall: {scores['avg.']['recall']:.4f}, "
              f"F1 Score: {scores['avg.']['f1']:.4f}, Accuracy: {scores['avg.']['accuracy']:.2f}%" 
              f"Total Correct Predictions: {sum(correct_predictions.values()) / 424}")

        print("\nPer-Category Statistics:")
        header = f"{'Category':<30} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Accuracy (%)':<15} "
        print(header)
        print("-" * len(header))
        for category in unique_categories:
            if category not in scores:
                continue
            cat_score = scores[category]
            print(f"{category:<30} {cat_score['precision']:.4f}    {cat_score['recall']:.4f}    "
              f"{cat_score['f1']:.4f}      {cat_score['accuracy']:.2f}%        "
              )

        scores_json[tokenizer_entry['tokenizer']] = scores

    # Save scores to JSON file
    with open(output_dir, 'w', encoding='utf-8') as json_file:
        json.dump(scores_json, json_file, ensure_ascii=False, indent=4)
        print(f"\nAll scores saved to {output_dir}")

if __name__ == '__main__':
    main()
