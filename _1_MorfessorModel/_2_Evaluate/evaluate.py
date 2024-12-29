import os

eval_path = '_1_MorfessorModel/Data/evaluation/71_Words.txt'
target_data = []
categories = []
current_category = ''

for line in open(eval_path):
    tokenizer = line.strip().split('\t')
    if len(tokenizer) == 1:
        current_category = tokenizer[0]
        continue
    if tokenizer[1][0] == '[':
        target = tokenizer[0].strip()
    else:
        target = tokenizer[1].split(' ')[0].strip()
    target_data.append(target.split('-'))
    categories.append(current_category)

def generate_spans(morphemes):
    spans_set = set()
    start = 0
    for morpheme in morphemes:
        stop = start + len(morpheme)
        spans_set.add((start, stop))
        start = stop
    return spans_set

def generate_pred_spans(prediction):
    if ' ' in prediction:
        return generate_spans(prediction.strip().split(' '))
    elif '-' in prediction:
        return generate_spans(prediction.strip().split('-'))
    else:
        return generate_spans([prediction.strip()])

def calculate_f1(size_overlap, size_target, size_pred):
    precision = size_overlap / size_pred if size_pred != 0 else 0.0
    recall = size_overlap / size_target if size_target != 0 else 0.0
    if recall == 0.0 or precision == 0.0:
        return 0.0
    else:
        return 2 * ((precision * recall) / (precision + recall))
            
def analyse_results(tokenizer_name):
    size_target = {}
    size_pred = {}
    size_overlap = {}

    if not os.path.isfile(tokenizer_name):
        print('Predictions not found:', tokenizer_name)
        return None

    out_file = open(tokenizer_name)

    for target, prediction, current_category in zip(target_data, out_file, categories):
        target_spans = generate_spans(target)
        prediction_spans = generate_pred_spans(prediction.strip().replace('#', ''))
        overlap = len(target_spans.intersection(prediction_spans))
        
        if current_category not in size_target:
            size_target[current_category] = 0
            size_pred[current_category] = 0
            size_overlap[current_category] = 0

        size_target[current_category] += len(target_spans)
        size_pred[current_category] += len(prediction_spans)
        size_overlap[current_category] += overlap
   
    results = []
    for current_category in sorted(size_target):
        f1 = calculate_f1(
            size_overlap=size_overlap[current_category], 
            size_target=size_target[current_category], 
            size_pred=size_pred[current_category]
        )
        results.append('{:.2f}'.format(100 * f1))
    return results[0]