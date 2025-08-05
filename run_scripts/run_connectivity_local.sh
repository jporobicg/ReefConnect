#!/bin/bash

# Configuration
SHAPEFILE_PATH="/home/por07g/Documents/Projects/GBR_modeling/ocean_parcels_gbr/data/Shape_files/gbr1_coral_1m_merged_buffer0p001.shp"
OUTPUT_DIR="/home/por07g/Documents/Projects/GBR_modeling/ocean_parcels_gbr/MRE/outputs/"
LOG_DIR="${OUTPUT_DIR}/logs"
SCRIPT_PATH="/home/por07g/Documents/Projects/GBR_modeling/ReefConnect/run_scripts/run_connectivity.py"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Array of release dates (YYYY-MM-DD format)
RELEASE_DATES=(
    "2015-10-29"
    "2015-11-28"
    "2015-12-27"
    "2016-10-18"
    "2016-11-17"
    "2016-12-16"
    "2017-10-08"
    "2017-11-06"
    "2017-12-06"
    "2018-10-27"
    "2018-11-25"
    "2018-12-25"
    "2019-10-16"
    "2019-11-15"
    "2019-12-14"
    "2020-10-04"
    "2020-11-03"
    "2020-12-02"
    "2021-01-01"
    "2021-10-23"
    "2021-11-21"
    "2021-12-21"
)

# Function to run connectivity analysis for a single date
run_connectivity() {
    local date=$1
    local log_file="${LOG_DIR}/connectivity_${date}.log"
    
    echo "Processing release date: $date"
    echo "Log file: $log_file"
    
    # Run the analysis and capture both stdout and stderr
    python "$SCRIPT_PATH" "$date" > "$log_file" 2>&1
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed $date"
    else
        echo "Error processing $date. Check log file: $log_file"
    fi
}

# Main execution
echo "Starting connectivity analysis for ${#RELEASE_DATES[@]} dates"
echo "Logs will be saved in: $LOG_DIR"

# Process each date
for date in "${RELEASE_DATES[@]}"; do
    run_connectivity "$date"
done

echo "All processing complete. Check logs in $LOG_DIR for details."


