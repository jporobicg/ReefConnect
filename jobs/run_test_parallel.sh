#!/bin/bash

# Test Parallel Connectivity Analysis
# ==================================

echo "=========================================="
echo "SUBMITTING PARALLEL CONNECTIVITY TEST JOB"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Submit the test job
echo "Submitting test job..."
jobid=$(sbatch --parsable jobs/test_parallel_job.sh)

if [ $? -eq 0 ]; then
    echo "✅ Test job submitted successfully!"
    echo "Job ID: $jobid"
    echo ""
    echo "Job details:"
    echo "  - Job ID: $jobid"
    echo "  - Output log: logs/test_parallel_${jobid}_0.out"
    echo "  - Error log: logs/test_parallel_${jobid}_0.err"
    echo ""
    echo "To monitor the job:"
    echo "  squeue -j $jobid"
    echo ""
    echo "To view stat details:"
    echo "  seff  $jobid"
    echo ""
    echo "To view output in real-time:"
    echo "  tail -f logs/test_parallel_${jobid}_0.out"
    echo ""
    echo "To view errors:"
    echo "  tail -f logs/test_parallel_${jobid}_0.err"
    echo ""
    echo "=========================================="
    echo "Test job submitted at: $(date)"
    echo "=========================================="
else
    echo "❌ Failed to submit test job!"
    exit 1
fi
