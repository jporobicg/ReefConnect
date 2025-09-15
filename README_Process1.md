# Process 1: Generate Connectivity Data with Bootstrap Resampling

## Overview

This module implements Process 1 of the ReefConnect tool, which calculates connectivity matrices between reefs in the Great Barrier Reef using particle tracking data and bootstrap resampling.

## Key Features

- **Uses Original Code Structure**: Maintains the original logic from `matrix_calculations.py` and `angle.py`
- **Bootstrap Resampling**: Samples 100 particles (without replacement) and repeats 50 times
- **Spatial Metrics**: Calculates angles, distances, and directions between all reef pairs
- **NetCDF Output**: Saves results in the specified NetCDF structure

## File Structure

```
ReefConnect/
├── original_code/              # Original code files
│   ├── matrix_calculations.py  # Original connectivity calculations
│   ├── angle.py               # Original spatial calculations
│   ├── get_kernels.py         # Original kernel calculations
│   └── Run_connectivity.sh    # Original shell script
└── process1_connectivity.py   # Process 1 implementation
```

## Usage

### Basic Usage
```bash
python process1_connectivity.py --shapefile path/to/reefs.shp --particle-data path/to/particle/files --output connectivity_data.nc
```

### Advanced Usage
```bash
python process1_connectivity.py \
    --shapefile /path/to/gbr1_coral_1m_merged_buffer0p001.shp \
    --particle-data /path/to/OceanParcels/outputs/Coral/ \
    --output connectivity_data.nc \
    --num-samples 100 \
    --num-repetitions 50
```

## Input Requirements

### Shapefile
- Must contain reef polygons with 'FID' field
- Should contain 3806 reef sites
- Format: ESRI Shapefile (.shp)

### Particle Data
- NetCDF files with particle tracking data
- Expected naming convention: `GBR1_H2p0_Coral_Release_*_Polygon_{reef_id}_Wind_3_percent_displacement_field.nc`
- Each file contains 1000 particle tracks for one source reef
- Variables: lat, lon, trajectory, age

## Output Structure

### NetCDF File
- **Dimensions**:
  - `source`: 3806 (source reefs)
  - `sink`: 3806 (sink reefs)
  - `sample`: 50 (bootstrap samples)

- **Variables**:
  - `angle[source, sink]`: Angle between source and sink reefs (degrees)
  - `distance[source, sink]`: Distance between source and sink reefs (kilometers)
  - `direction[source, sink]`: Directional sector from source to sink (0-35)
  - `connectivity[source, sink, sample]`: Bootstrap connectivity values (probability)

## Implementation Status

### ✅ Completed
- File structure and basic framework
- Spatial metrics calculation (angles, distances, directions)
- Bootstrap resampling framework
- NetCDF output generation
- Particle file loading and processing
- Progress tracking and error handling

### 🔄 In Progress
- Integration with original `calc()` function
- Full connectivity calculation using decay and competence functions
- Proper particle settlement detection

### 📋 TODO
- Complete integration with original `calc()` function logic
- Implement proper particle-in-polygon detection for each sink reef
- Add decay and competence function calculations
- Optimize for large datasets (3806 reefs)
- Add parallel processing capabilities

## Dependencies

- `numpy`: Numerical computations
- `xarray`: NetCDF file handling
- `geopandas`: Geographic data processing
- `pandas`: Data manipulation
- `tqdm`: Progress bars
- `scipy`: Scientific computing (for original functions)

## Original Code Integration

This implementation preserves the original code structure and logic:

1. **Spatial Calculations**: Uses original `angle()`, `haversine()`, and `veclength()` functions
2. **Connectivity Calculations**: Integrates with original `calc()` function framework
3. **Parameters**: Uses original decay and competence parameters from `matrix_calculations.py`
4. **File Handling**: Follows original particle file naming and loading conventions

## Notes

- The current implementation includes placeholder connectivity calculations
- Full integration with the original `calc()` function is needed for production use
- The bootstrap resampling framework is complete and ready for integration
- Spatial metrics calculation is fully functional using original logic

## Next Steps

1. Complete integration with original `calc()` function
2. Test with actual particle data
3. Optimize performance for 3806 reefs
4. Add Process 2 implementation for kernel estimation 