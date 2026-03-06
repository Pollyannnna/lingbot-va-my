#!/bin/bash

# Source and destination directories
SRC_DIR="/data/250010187/yeziyang1/RoboTwin/results/full_eval_0303/stseed-10000/visualization"
DEST_DIR="/data/250010187/yeziyang1/RoboTwin/results/false"

# Create destination directory
mkdir -p "$DEST_DIR"

# Find all False mp4 files and move them preserving directory structure
echo "Finding all False video files..."
find "$SRC_DIR" -type f -name "*False*.mp4" | while read -r file; do
    # Get the relative path from the source directory
    rel_path="${file#$SRC_DIR/}"
    
    # Get the directory part
    dir_part=$(dirname "$rel_path")
    
    # Create the destination directory if it doesn't exist
    mkdir -p "$DEST_DIR/$dir_part"
    
    # Move the file
    mv "$file" "$DEST_DIR/$dir_part/"
    
    echo "Moved: $rel_path"
done

echo ""
echo "Done! All False videos have been moved to $DEST_DIR"
echo "Directory structure has been preserved."
