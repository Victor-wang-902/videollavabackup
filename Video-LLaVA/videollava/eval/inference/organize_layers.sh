#!/bin/bash

# Define the layers you're interested in
layers=(1 4 8 16 24 31)

# Loop over each layer
for layer in "${layers[@]}"; do
    # Create a directory for each layer if it doesn't exist
    mkdir -p "finetuned-ours_layer_$layer"

    # Loop over each pretrained directory
    for dir in finetuned-ours-*; do
        # Define the source file path
        src="$dir/attn_maps/atten_map_layer${layer}.png"

        # Check if the source file exists
        if [ -f "$src" ]; then
            # Define the destination file path with a unique name
            dest="finetuned-ours_layer_${layer}/atten_map_layer${layer}_${dir}.png"

            # Copy the file to the destination directory
            cp "$src" "$dest"
        else
            echo "File $src does not exist."
        fi
    done
done

echo "Files have been organized successfully."
