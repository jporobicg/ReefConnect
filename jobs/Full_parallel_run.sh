#!/bin/bash
#SBATCH --job-name=full_connectivity
#SBATCH --output=logs/full_connectivity_%A_%a.out
#SBATCH --error=logs/full_connectivity_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=64
#SBATCH --array=0-499
#SBATCH --partition=compute

# Calculate which run, chunk, and species we're processing
# Each job will do 10 repetitions (set in config file)
NUM_CHUNKS=10
NUM_SPECIES=2

run_idx=$((SLURM_ARRAY_TASK_ID / (NUM_CHUNKS * NUM_SPECIES)))
chunk_num=$(((SLURM_ARRAY_TASK_ID / NUM_SPECIES) % NUM_CHUNKS))
species_idx=$((SLURM_ARRAY_TASK_ID % NUM_SPECIES))

# Print job information
echo "=========================================="
echo "PARALLEL CONNECTIVITY TEST JOB"
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
    "2015-10-29" "2015-11-05" "2015-11-12" "2015-11-19" "2015-11-26"
    "2015-12-03" "2015-12-10" "2015-12-17" "2015-12-24" "2015-12-31"
    "2016-01-07" "2016-01-14" "2016-01-21" "2016-01-28" "2016-02-04"
    "2016-02-11" "2016-02-18" "2016-02-25" "2016-03-03" "2016-03-10"
    "2016-03-17" "2016-03-24" "2016-03-31" "2016-04-07"
)

# Define species array
species=("acropora" "merulinidae")

# Change to project directory (use relative path)
cd "$(dirname "$0")/.." || {
    echo "Error: Cannot change to project directory"
    exit 1
}

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
