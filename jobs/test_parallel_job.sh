#!/bin/bash
#SBATCH --account=OD-232538
#SBATCH --time=2:00:00
#SBATCH --mem=50g
#SBATCH --cpus-per-task=8
#SBATCH --array=0-0%1  # Single job for testing
#SBATCH --output=logs/test_parallel_%A_%a.out
#SBATCH --error=logs/test_parallel_%A_%a.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load Python module
module load python

# Set environment variables for parallel processing
export SLURM_CPUS_ON_NODE=8

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

# Change to project directory
cd /datasets/work/oa-coconet/work/ReefConnect/
# cd /home/por07g/Documents/Projects/GBR_modeling/ReefConnect

# Activate conda environment
# conda activate GBR_env

# Test with small chunk size and limited repetitions
echo "Running parallel connectivity analysis with test parameters..."
echo "Chunk size: 1 (for testing)"
echo "Max repetitions: 2 (for testing)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

# Run the parallel script with test parameters
python run_connectivity_parallel.py \
    --config config/connectivity_parameters.yaml \
    --release-day 2015-10-29 \
    --chunk-size 200 \
    --output output/test_parallel_results.nc

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "TEST JOB COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "End time: $(date)"
    echo "Output file: output/test_parallel_results.nc"
    
    # Show file size
    if [ -f "output/test_parallel_results.nc" ]; then
        file_size=$(du -h output/test_parallel_results.nc | cut -f1)
        echo "Output file size: $file_size"
    fi
    
    # Show memory usage summary
    echo ""
    echo "Memory usage summary:"
    echo "Peak memory: $(grep 'Peak memory' logs/test_parallel_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out | tail -1 || echo 'Not available')"
    
else
    echo ""
    echo "=========================================="
    echo "TEST JOB FAILED"
    echo "=========================================="
    echo "End time: $(date)"
    echo "Check error log: logs/test_parallel_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
    exit 1
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
