#!/bin/bash

# List of folders to iterate through
folders=(
    "data/doc2dial/new_dataset"
    "data/doc2dial/new_fid"
    # Add more folder paths here
)
python_script="extra_eval.py"

# Iterate through each folder in the list
for folder in "${folders[@]}"; do
    python3 "$python_script" -t "$folder"
done