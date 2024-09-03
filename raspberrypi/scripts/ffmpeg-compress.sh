#!/bin/bash

# Define the directory
DIRECTORY='/media/hdddisk/movie'

# Check if the provided argument is a directory
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: $DIRECTORY is not a directory."
    exit 1
fi

cd "$DIRECTORY"

# Traverse the directory and compress the files
echo "Traversing directory: $DIRECTORY"
for FILE in "$DIRECTORY"/*; do
    if [ -f "$FILE" ]; then
        FILE_NEW="${FILE%.mp4}-compressed.mp4"
        echo "compressing $FILE"
        ffmpeg -i "$FILE" -c:v libx264 -tag:v avc1 -movflags faststart -crf 30 -preset superfast "$FILE_NEW"
    fi
done