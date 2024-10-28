#!/bin/bash

# File Path
FILES=("MARBLES.BMP" "RAY.BMP" "big.txt")
OUTPUT_FILE="compression_results.csv"

# Create or overwrite the output file with headers
echo "File,Level,OriginalSize,CompressedSize,CompressionRate,TimeTaken" > $OUTPUT_FILE

for FILE in "${FILES[@]}"; do
  for LEVEL in {1..9}; do  # Gzip levels go from 1 to 9
    # Get the size of the original file
    ORIGINAL_SIZE=$(stat -c%s "$FILE")

    # Measure the compression time
    START_TIME=$(date +%s.%N)
    gzip -c -$LEVEL "$FILE" > "${FILE}_${LEVEL}.gz"
    END_TIME=$(date +%s.%N)

    # Calculate the compression time
    TIME_TAKEN=$(echo "$END_TIME - $START_TIME" | bc)

    # Get the size of the compressed file
    COMPRESSED_SIZE=$(stat -c%s "${FILE}_${LEVEL}.gz")

    # Calculate the compression rate
    COMPRESSION_RATE=$(echo "scale=2; $COMPRESSED_SIZE / $ORIGINAL_SIZE * 100" | bc)

    # Append the results to the CSV file
    echo "$FILE,$LEVEL,$ORIGINAL_SIZE,$COMPRESSED_SIZE,$COMPRESSION_RATE,$TIME_TAKEN" >> $OUTPUT_FILE

    # Remove the compressed file to save space
    rm "${FILE}_${LEVEL}.gz"
  done
done
