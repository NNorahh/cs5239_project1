#!/bin/bash

# File Paths
FILES=("MARBLES.BMP" "RAY.BMP" "big.txt")
OUTPUT_FILE="compression_results.csv"
ARCHIVE_NAME="combined_files.tar"

# Create or overwrite the output file with headers
echo "File,Level,OriginalSize,CompressedSize,CompressionRate,TimeTaken" > $OUTPUT_FILE

# Create a tar archive containing all files
tar -cf $ARCHIVE_NAME "${FILES[@]}"

# Get the size of the original archive
ORIGINAL_SIZE=$(stat -c%s "$ARCHIVE_NAME")

# Loop through compression levels
for LEVEL in {1..9}; do  # Gzip levels go from 1 to 9
    # Measure the compression time
    START_TIME=$(date +%s.%N)
    gzip -c -$LEVEL "$ARCHIVE_NAME" > "${ARCHIVE_NAME}_${LEVEL}.tar.gz"
    END_TIME=$(date +%s.%N)

    # Calculate the compression time
    TIME_TAKEN=$(echo "$END_TIME - $START_TIME" | bc)

    # Get the size of the compressed archive
    COMPRESSED_SIZE=$(stat -c%s "${ARCHIVE_NAME}_${LEVEL}.tar.gz")

    # Calculate the compression rate
    COMPRESSION_RATE=$(echo "scale=2; $COMPRESSED_SIZE / $ORIGINAL_SIZE * 100" | bc)

    # Append the results to the CSV file
    echo "$ARCHIVE_NAME,$LEVEL,$ORIGINAL_SIZE,$COMPRESSED_SIZE,$COMPRESSION_RATE,$TIME_TAKEN" >> $OUTPUT_FILE

    # Remove the compressed archive to save space
    rm "${ARCHIVE_NAME}_${LEVEL}.tar.gz"
done

# Clean up the original archive
rm $ARCHIVE_NAME
