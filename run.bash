#!/bin/bash

config_file="config.ini"
python_script="run_exp.py"

# Get section names from the config file
# sections=$(awk -F'[][]' '/^\[.*\]$/ && !/\[DEFAULT\]/ { print $2 }' "$config_file")
sections=$(awk -F'[][]' '/^\[.*\]$/{ print $2 }' "$config_file")

# Iterate over each section and run the Python script
for section in $sections; do
    # if [ "$section" != "DEFAULT" ]; then
        echo "Running section: $section"

  # Run the Python script with the section name as an argument
      python3 "$python_script" --config "$section"

    echo  # Add a newline for separation between sections
    # fi
done






