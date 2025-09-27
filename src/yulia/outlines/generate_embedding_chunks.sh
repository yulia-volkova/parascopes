#!/bin/bash

BASE_DIR="/workspace/ALGOVERSE/yas/yulia/parascopes/src"
cd "$BASE_DIR"

# Export Python path
export PYTHONPATH=$PYTHONPATH:$BASE_DIR

for chunk in {1..9}; do
    echo "Processing chunk $chunk..."
    python yulia/outlines/generate_outline_embeddings.py $chunk
    
    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error processing chunk $chunk. Stopping."
        exit 1
    fi
    
    echo "Completed chunk $chunk"
    echo "-------------------"
done

echo "All chunks completed!"
