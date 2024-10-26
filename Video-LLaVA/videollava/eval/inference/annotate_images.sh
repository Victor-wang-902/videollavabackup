#!/bin/bash

# Loop over each pretrained_layer_* directory
for layer_dir in finetuned-ours_layer_*; do
    echo "Processing directory $layer_dir"
    # Create 'annotated' subdirectory if it doesn't exist
    mkdir -p "$layer_dir/annotated"
    # Loop over each PNG file in the directory
    for img_file in "$layer_dir"/*.png; do
        echo "Processing file $img_file"
        # Extract the checkpoint number from the filename
        filename=$(basename "$img_file")
        filename_no_ext="${filename%.*}"
        if [[ $filename_no_ext =~ (pretrained|finetuned-ours)-([0-9]+) ]]; then
            checkpoint="${BASH_REMATCH[2]}"
            echo "Checkpoint: $checkpoint"
            # Define the output file path
            output_file="$layer_dir/annotated/$filename"
            # Annotate the image using ImageMagick
            convert "$img_file" \
                -gravity North \
                -pointsize 100 \
                -stroke '#000C' -strokewidth 2 -annotate +0+10 "$checkpoint" \
                -stroke none -fill white -annotate +0+10 "$checkpoint" \
                "$output_file"
        else
            echo "Could not extract checkpoint number from filename $filename"
        fi
    done
done

echo "All images have been annotated and saved in the 'annotated' directories."
