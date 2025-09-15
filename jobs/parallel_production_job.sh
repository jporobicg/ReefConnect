#!/bin/bash
#SBATCH --account=OD-232538
#SBATCH --time=24:00:00
#SBATCH --mem=200g
#SBATCH --cpus-per-task=64
#SBATCH --array=0-24%25  # 25 date ranges, max 25 concurrent
#SBATCH --output=logs/parallel_prod_%A_%a.out
#SBATCH --error=logs/parallel_prod_%A_%a.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load Python module
module load python

# Set environment variables for parallel processing
export SLURM_CPUS_ON_NODE=64

# Define date ranges
dates=(
    "2015-10-29 T20:00 2015-11-02 T23:59"
    "2015-11-28 T20:00 2015-12-02 T23:59"
    "2015-12-27 T20:00 2015-12-31 T23:59"
    "2016-10-18 T20:00 2016-10-22 T23:59"
    "2016-11-17 T20:00 2016-11-21 T23:59"
    "2016-12-16 T20:00 2016-12-20 T23:59"
    "2017-10-08 T20:00 2017-10-12 T23:59"
    "2017-11-06 T20:00 2017-11-10 T23:59"
    "2017-12-06 T20:00 2017-12-10 T23:59"
    "2018-10-27 T20:00 2018-10-31 T23:59"
    "2018-11-25 T20:00 2018-11-30 T23:59"
    "2018-12-25 T20:00 2018-12-30 T23:59"
    "2019-10-16 T20:00 2019-10-20 T23:59"
    "2019-11-15 T20:00 2019-11-19 T23:59"
    "2019-12-14 T20:00 2019-12-18 T23:59"
    "2020-10-04 T20:00 2020-10-08 T23:59"
    "2020-11-03 T20:00 2020-11-07 T23:59"
    "2020-12-02 T20:00 2020-12-06 T23:59"
    "2021-01-01 T20:00 2021-01-05 T23:59"
    "2021-10-23 T20:00 2021-10-27 T23:59"
    "2021-11-21 T20:00 2021-11-25 T23:59"
    "2021-12-21 T20:00 2021-12-25 T23:59"
    "2022-10-09 T20:00 2022-10-13 T23:59"
    "2022-11-08 T20:00 2022-11-12 T23:59"
    "2022-10-07 T20:00 2022-10-11 T23:59"
)

# Get the date range for this job
date_range="${dates[$SLURM_ARRAY_TASK_ID]}"
start_date=$(echo $date_range | cut -d' ' -f1)

# Print job information
echo "=========================================="
echo "PARALLEL CONNECTIVITY PRODUCTION JOB"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Date range: $date_range"
echo "Start date: $start_date"
echo "Node: $SLURM_NODELIST"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="

# Change to project directory
cd /home/por07g/Documents/Projects/GBR_modeling/ReefConnect

# Activate conda environment
conda activate GBR_env

# Run the parallel script for this date range
echo "Running parallel connectivity analysis for date range: $date_range"
echo "Chunk size: 200"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo ""

python run_connectivity_parallel.py \
    --config config/connectivity_parameters.yaml \
    --release-day $start_date \
    --chunk-size 200 \
    --output output/connectivity_results_${start_date}.nc

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "PRODUCTION JOB COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "End time: $(date)"
    echo "Output file: output/connectivity_results_${start_date}.nc"
    
    # Show file size
    if [ -f "output/connectivity_results_${start_date}.nc" ]; then
        file_size=$(du -h output/connectivity_results_${start_date}.nc | cut -f1)
        echo "Output file size: $file_size"
    fi
    
else
    echo ""
    echo "=========================================="
    echo "PRODUCTION JOB FAILED"
    echo "=========================================="
    echo "End time: $(date)"
    echo "Check error log: logs/parallel_prod_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
    exit 1
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
