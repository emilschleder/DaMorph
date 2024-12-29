import os
import evaluate
import matplotlib.pyplot as plt
import numpy as np

def segment_words_with_models(models_folder):
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.txt')]
    model_results = {}

    for model_file in model_files:
        f1_score = float(evaluate.analyse_results(f'{models_folder}/{model_file}'))
        model_name = model_file.replace('segmented_', '').replace('.txt', '')
        model_results[model_name] = f1_score

    # Sorting function 
    def sort_key(name):
        base_name = name
        for description_name in ['_All_10mil', '_All_2mil', '_All_400_000', '_COR']:
            base_name = base_name.replace(description_name, '')
        return int(base_name.replace('Model', ''))
    
    sorted_results = dict(sorted(model_results.items(), key=lambda x: sort_key(x[0])))
    plot_results(sorted_results)

def plot_results(results):
    # Prepare data for plotting
    unique_models = sorted(
        set(
            name.replace('_All_10mil', '')
                .replace('_All_2mil', '')
                .replace('_All_400_000', '')
                .replace('_COR', '')
            for name in results.keys()
        ),
        key=lambda x: int(x.replace('Model', ''))
    )
    # placeholders for: [COR, 10mil, 2mil, 400_000]
    grouped_scores = {model: [None, None, None, None] for model in unique_models}  
    dataset_indices = {'COR': 0, '10mil': 1, '2mil': 2, '400_000': 3}

    
    for model_name, f1_score in results.items():
        if '_All_10mil' in model_name:
            base_name = model_name.replace('_All_10mil', '')
            grouped_scores[base_name][dataset_indices['10mil']] = f1_score
        elif '_All_2mil' in model_name:
            base_name = model_name.replace('_All_2mil', '')
            grouped_scores[base_name][dataset_indices['2mil']] = f1_score
        elif '_All_400_000' in model_name:
            base_name = model_name.replace('_All_400_000', '')
            grouped_scores[base_name][dataset_indices['400_000']] = f1_score
        else:
            base_name = model_name.replace('_COR', '')
            grouped_scores[base_name][dataset_indices['COR']] = f1_score

    # Prepare data for plotting
    labels = [model.replace('Model', '') for model in unique_models]
    model_numbers = [int(label) for label in labels]
    heights_COR = []
    heights_10mil = []
    heights_2mil = []
    heights_400k = []
    
    for model in unique_models:
        heights_COR.append(
            grouped_scores[model][dataset_indices['COR']] if grouped_scores[model][dataset_indices['COR']] is not None else 0
        )
        heights_10mil.append(
            grouped_scores[model][dataset_indices['10mil']] if grouped_scores[model][dataset_indices['10mil']] is not None else 0
        )
        heights_2mil.append(
            grouped_scores[model][dataset_indices['2mil']] if grouped_scores[model][dataset_indices['2mil']] is not None else 0
        )
        heights_400k.append(
            grouped_scores[model][dataset_indices['400_000']] if grouped_scores[model][dataset_indices['400_000']] is not None else 0
        )

    line_colors = [
        '#8FBC8F',
        '#d97d7d',
        '#F0E68C',
        '#779dc3'
    ]

    plt.figure(figsize=(12, 6))
    plt.plot(model_numbers, heights_COR, marker='o', color=line_colors[0], label='COR Corpus')
    plt.plot(model_numbers, heights_10mil, marker='o', color=line_colors[1], label='Unique Words Corpus 10 mil')
    plt.plot(model_numbers, heights_2mil, marker='o', color=line_colors[2], label='Unique Words Corpus 2 mil')
    plt.plot(model_numbers, heights_400k, marker='o', color=line_colors[3], label='Unique Words Corpus 400,000')

    plt.xlabel('Annotated Data Size', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score by Model and Corpus', fontsize=14)
    plt.xticks(model_numbers, labels, rotation=45)
    plt.legend()
    plt.grid(visible=True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
Pred_folder = '_1_MorfessorModel/Preds'
segment_words_with_models(Pred_folder)