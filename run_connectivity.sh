#!/bin/bash

# run_connectivity.sh - Bash script to run reef connectivity analysis
# 
# This script iterates over all particle output files in a given folder
# and calls the connectivity analysis for each file with necessary arguments

# Default values
RELEASE_DAY=""
PARTICLE_DATA_DIR=""
SHAPEFILE_PATH=""
OUTPUT_DIR=""
CONFIG_FILE="config/connectivity_parameters.yaml"
PYTHON_SCRIPT="process1_connectivity.py"

# Help function
show_help() {
    echo "Usage: $0 -d RELEASE_DAY -p PARTICLE_DATA_DIR -s SHAPEFILE_PATH -o OUTPUT_DIR [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  -d RELEASE_DAY       Release day string (e.g., '365')"
    echo "  -p PARTICLE_DATA_DIR Directory containing particle NetCDF files"
    echo "  -s SHAPEFILE_PATH    Path to reef shapefile"
    echo "  -o OUTPUT_DIR        Output directory for results"
    echo ""
    echo "Optional arguments:"
    echo "  -c CONFIG_FILE       Path to YAML configuration file (default: config/connectivity_parameters.yaml)"
    echo "  -h                   Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -d 365 -p /path/to/particle/data -s /path/to/reefs.shp -o ./results"
}

# Parse command line arguments
while getopts "d:p:s:o:c:h" opt; do
    case $opt in
        d) RELEASE_DAY="$OPTARG" ;;
        p) PARTICLE_DATA_DIR="$OPTARG" ;;
        s) SHAPEFILE_PATH="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        c) CONFIG_FILE="$OPTARG" ;;
        h) show_help; exit 0 ;;
        *) echo "Invalid option: -$OPTARG" >&2; show_help; exit 1 ;;
    esac
done

# Check required arguments
if [[ -z "$RELEASE_DAY" || -z "$PARTICLE_DATA_DIR" || -z "$SHAPEFILE_PATH" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: Missing required arguments."
    show_help
    exit 1
fi

# Check if directories and files exist
if [[ ! -d "$PARTICLE_DATA_DIR" ]]; then
    echo "Error: Particle data directory does not exist: $PARTICLE_DATA_DIR"
    exit 1
fi

if [[ ! -f "$SHAPEFILE_PATH" ]]; then
    echo "Error: Shapefile does not exist: $SHAPEFILE_PATH"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file does not exist: $CONFIG_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
echo "Activating conda environment: GBR_env"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate GBR_env

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script does not exist: $PYTHON_SCRIPT"
    exit 1
fi

# Find all particle files for the given release day
echo "Searching for particle files in: $PARTICLE_DATA_DIR"
PATTERN="GBR1_H2p0_Coral_Release_${RELEASE_DAY}_Polygon_*_Wind_3_percent_displacement_field.nc"
PARTICLE_FILES=($(find "$PARTICLE_DATA_DIR" -name "$PATTERN" | sort))

if [[ ${#PARTICLE_FILES[@]} -eq 0 ]]; then
    echo "Error: No particle files found matching pattern: $PATTERN"
    exit 1
fi

echo "Found ${#PARTICLE_FILES[@]} particle files for release day $RELEASE_DAY"

# Output file path
OUTPUT_FILE="$OUTPUT_DIR/connectivity_results_${RELEASE_DAY}.nc"

echo "Starting connectivity analysis..."
echo "Release day: $RELEASE_DAY"
echo "Particle data directory: $PARTICLE_DATA_DIR"
echo "Shapefile: $SHAPEFILE_PATH"
echo "Configuration: $CONFIG_FILE"
echo "Output file: $OUTPUT_FILE"
echo "================================"

# Record start time
START_TIME=$(date +%s)

# Run the Python script
python "$PYTHON_SCRIPT" \
    --release_day "$RELEASE_DAY" \
    --particle_data_dir "$PARTICLE_DATA_DIR" \
    --shapefile_path "$SHAPEFILE_PATH" \
    --output_file "$OUTPUT_FILE" \
    --config_file "$CONFIG_FILE"

# Check if the script ran successfully
if [[ $? -eq 0 ]]; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo "================================"
    echo "✅ Connectivity analysis completed successfully!"
    echo "Duration: ${DURATION} seconds"
    echo "Output saved to: $OUTPUT_FILE"
    
    # Verify output file
    if [[ -f "$OUTPUT_FILE" ]]; then
        FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
        echo "Output file size: $FILE_SIZE"
    fi
else
    echo ""
    echo "================================"
    echo "❌ Connectivity analysis failed!"
    exit 1
fi 