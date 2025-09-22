# Connectivity Matrix Comparison Documentation

## Overview

This directory contains comprehensive documentation and tools for comparing connectivity matrices from different analysis runs, focusing on validating the new parallel processing system against historical outputs.

## Files

### Documentation
- **`connectivity_comparison_analysis.md`** - Detailed analysis of current comparison results and methodology
- **`connectivity_comparison_todo.md`** - Comprehensive TODO list for future comparisons
- **`README.md`** - This overview file

### Scripts
- **`track_comparison_progress.py`** - Track progress on TODO list tasks
- **`update_todo_status.py`** - Update individual task statuses

## Current Status

### ✅ Completed
- Initial comparison between CSV (2015-10-29) and NetCDF (test output)
- New system validation (NetCDF structure and data quality)
- Documentation framework creation
- TODO list establishment

### 🔄 In Progress
- Historical data cataloging
- Automated comparison tool development

### 📋 Next Steps
1. Run new system for all 25 historical dates
2. Complete systematic comparisons
3. Document findings and patterns
4. Create comprehensive validation report

## Key Findings

### CSV vs NetCDF Comparison (2015-10-29)
- **Different matrix dimensions**: CSV (3805×3806) vs NetCDF (3806×3806)
- **Different connectivity values**: CSV values 10-16× higher than NetCDF
- **Different sparsity**: CSV 96.56% vs NetCDF 97.89%
- **Low correlation**: 0.406 (expected due to different processing methods)

### New System Validation
- ✅ Correct 4D structure (3806×3806×2 treatments×100 samples)
- ✅ Both treatments working (Moneghetti and Connolly)
- ✅ Biologically realistic connectivity values
- ✅ Appropriate sparsity for marine larval dispersal

## Usage

### Track Progress
```bash
python scripts/track_comparison_progress.py
```

### Update Task Status
```bash
# List available tasks
python scripts/update_todo_status.py --list

# Update a specific task
python scripts/update_todo_status.py "catalog all historical CSV files" "✅"
```

### Available Status Options
- `✅` - Completed
- `🔄` - In Progress
- `⏸️` - Paused
- `❌` - Cancelled
- `` (empty) - Not Started

## Future Work

The TODO list contains 100+ tasks organized into:
- **Immediate tasks** (1-2 weeks)
- **High priority comparisons** (1 month)
- **Medium priority comparisons** (2-3 months)
- **Low priority comparisons** (future work)

Key focus areas:
1. Complete historical date comparisons (25 dates)
2. Species-specific analysis (Acropora, Merulinidae)
3. Treatment-specific validation (Moneghetti, Connolly)
4. Parameter sensitivity analysis
5. Spatial pattern analysis

## Contact

For questions about this documentation or the comparison process, refer to the detailed analysis in `connectivity_comparison_analysis.md`.

---

*Last Updated: September 22, 2024*
