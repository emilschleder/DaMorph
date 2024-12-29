#!/bin/bash

file_paths=(
    "_1_MorfessorModel/Data/corpus/COR_419_472.txt"
    "_1_MorfessorModel/Data/corpus/Most_Frequent_400_000.txt"
    "_1_MorfessorModel/Data/corpus/Most_Frequent_2_000_000.txt"
    "_1_MorfessorModel/Data/corpus/Most_Frequent_10_000_000.txt"
)

python_script="_1_MorfessorModel/_1_Training/train.py"
data_path="_1_MorfessorModel/Data/annotations"
output_model_path="_1_MorfessorModel/Models"

# Loop through each file path
for file_path in "${file_paths[@]}"; do
    echo "Processing file: $file_path"
    python "$python_script" --data_path "$data_path" --output_model_path "$output_model_path" --corpus "$file_path"
done