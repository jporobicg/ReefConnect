#!/bin/bash

# Load required modules (adjust according to your system)
module load python

# Define paths
SHAPEFILE_PATH="/datasets/work/oa-coconet/work/oceanparcels_gbr_Coral/Shape_files/gbr1_coral_1m_merged_buffer0p001.shp"
OUTPUT_DIR="/datasets/work/oa-coconet/work/Outputs_new_Runs/Angles"

# Run the scripts
echo "Starting ReefConnect analysis..."

echo "Processing reef angles..."
python /datasets/work/oa-coconet/work/ReefConnect/reefconnect/scripts/get_angles.py \
    --shapefile $SHAPEFILE_PATH \
    --output-dir $OUTPUT_DIR \
    --verbose

echo "Processing reef connectivity..."
#python /datasets/work/oa-coconet/work/ReefConnect/reefconnect/scripts/get_connectivity.py \
#    --shapefile $SHAPEFILE_PATH \
#    --output-dir $OUTPUT_DIR

echo "Processing reef kernels..."
#python /datasets/work/oa-coconet/work/ReefConnect/reefconnect/scripts/get_kernels.py \
#    --shapefile $SHAPEFILE_PATH \
#    --output-dir $OUTPUT_DIR

echo "Analysis complete!"
