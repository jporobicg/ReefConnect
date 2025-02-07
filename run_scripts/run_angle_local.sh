#! /bin/bash

SHAPEFILE_PATH="/home/por07g/Documents/Projects/GBR_modeling/ocean_parcels_gbr/data/Shape_files/gbr1_coral_1m_merged_buffer0p001.shp"
OUTPUT_DIR="/home/por07g/Documents/Projects/GBR_modeling/New_outputs/Angles/"

python /home/por07g/Documents/Projects/GBR_modeling/ReefConnect/reefconnect/scripts/get_angles.py --shapefile $SHAPEFILE_PATH --output-dir $OUTPUT_DIR --verbose


