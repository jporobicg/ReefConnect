#!/bin/bash
#SBATCH --account=OD-232538
#SBATCH --job-name=full_connectivity
#SBATCH --output=logs/full_connectivity_%A_%a.out
#SBATCH --error=logs/full_connectivity_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=64
#SBATCH --array=0-439
##SBATCH --partition=compute

# Calculate which run, chunk, and species we're processing

NUM_CHUNKS=10
NUM_SPECIES=2

run_idx=$((SLURM_ARRAY_TASK_ID / (NUM_CHUNKS * NUM_SPECIES)))
chunk_num=$(((SLURM_ARRAY_TASK_ID / NUM_SPECIES) % NUM_CHUNKS))
species_idx=$((SLURM_ARRAY_TASK_ID % NUM_SPECIES))

# Create logs directory if it doesn't exist
mkdir -p logs

# Load Python module
module load python
# Print job information
echo "=========================================="
echo "PARALLEL CONNECTIVITY Full JOB"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="
# Activate conda environment
# conda activate GBR_env

# Show processing parameters
echo "Running parallel connectivity analysis..."
echo "Chunk size: 200"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""


# Define date array (25 different dates)
dates=(
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

# Define species array
species=("acropora" "merulinidae")

# Change to project directory (use relative path)
cd /datasets/work/oa-coconet/work/ReefConnect/
# Get the date and species for this chunk
date_model="${dates[$chunk_num]}"
current_species="${species[$species_idx]}"

# Create output filename
name_output="connectivity_${date_model}_${current_species}_run${run_idx}"

echo "Processing date: $date_model"
echo "Processing species: $current_species"
echo "Run index: $run_idx"
echo "Chunk number: $chunk_num"
echo "Species index: $species_idx"
echo "Output file: $name_output.nc"
echo ""

# Run the connectivity analysis
python run_connectivity_parallel.py \
  --config config/connectivity_parameters.yaml \
  --release-day "$date_model" \
  --species "$current_species" \
  --chunk-size 200 \
  --output "output/$name_output.nc"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "JOB COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "End time: $(date)"
    echo "Output file: output/$name_output.nc"
    
    # Show file size
    if [ -f "output/$name_output.nc" ]; then
        file_size=$(du -h "output/$name_output.nc" | cut -f1)
        echo "Output file size: $file_size"
    fi
    
else
    echo ""
    echo "=========================================="
    echo "JOB FAILED"
    echo "=========================================="
    echo "End time: $(date)"
    echo "Check error log: logs/full_connectivity_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
    exit 1
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
