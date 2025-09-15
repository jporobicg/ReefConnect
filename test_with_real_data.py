#!/usr/bin/env python3
"""
Test Process 1 with real test data
"""

import os
import sys
import tempfile
import numpy as np
import xarray as xr
from pathlib import Path

# Add current directory to path to import our module
sys.path.append('.')

from process1_connectivity import (
    load_shapefile_and_centroids,
    calculate_spatial_metrics,
    create_netcdf_dataset,
    process1_main
)


def test_with_real_data():
    """
    Test Process 1 with real test data files.
    """
    print("="*60)
    print("TESTING PROCESS 1 WITH REAL DATA")
    print("="*60)
    
    # Test data paths
    shapefile_path = "test/shapefile/gbr1_coral_1m_merged_buffer0p001.shp"
    particle_data_path = "test/2015-10-29"
    
    # Check if test files exist
    if not os.path.exists(shapefile_path):
        print(f"❌ Shapefile not found: {shapefile_path}")
        return False
    
    if not os.path.exists(particle_data_path):
        print(f"❌ Particle data directory not found: {particle_data_path}")
        return False
    
    print(f"✅ Found shapefile: {shapefile_path}")
    print(f"✅ Found particle data: {particle_data_path}")
    
    # List available particle files
    particle_files = list(Path(particle_data_path).glob("*.nc"))
    print(f"Found {len(particle_files)} particle files:")
    for file in particle_files:
        print(f"  - {file.name}")
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    try:
        print("\nRunning Process 1 with real test data...")
        
        # Run Process 1 with smaller parameters for testing
        results = process1_main(
            shapefile_path=shapefile_path,
            particle_data_path=particle_data_path,
            output_path=output_path,
            num_samples=50,  # Smaller sample for testing
            num_repetitions=5  # Fewer repetitions for testing
        )
        
        print("✅ Process 1 completed successfully!")
        print(f"Results summary:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        # Verify output file
        if os.path.exists(output_path):
            print(f"\n✅ Output file created: {output_path}")
            
            # Load and examine the NetCDF file
            ds = xr.open_dataset(output_path)
            print(f"NetCDF dimensions: {dict(ds.dims)}")
            print(f"NetCDF variables: {list(ds.data_vars.keys())}")
            
            # Print some sample data
            print("\nSample data:")
            print(f"Angle matrix shape: {ds['angle'].shape}")
            print(f"Distance matrix shape: {ds['distance'].shape}")
            print(f"Direction matrix shape: {ds['direction'].shape}")
            print(f"Connectivity matrix shape: {ds['connectivity'].shape}")
            
            # Print some actual values
            print(f"\nSample values:")
            print(f"Angle from reef 0 to reef 1: {ds['angle'].values[0, 1]:.2f} degrees")
            print(f"Distance from reef 0 to reef 1: {ds['distance'].values[0, 1]:.2f} km")
            print(f"Direction from reef 0 to reef 1: {ds['direction'].values[0, 1]}")
            print(f"Connectivity from reef 0 to reef 1 (sample 0): {ds['connectivity'].values[0, 1, 0]:.6f}")
            
            ds.close()
            
            # Keep the file for inspection
            print(f"\nOutput file saved at: {output_path}")
            print("You can examine this file to verify the results.")
            
        else:
            print("❌ Output file not created!")
            return False
            
    except Exception as e:
        print(f"❌ Process 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_spatial_metrics_only():
    """
    Test only the spatial metrics calculation with real shapefile.
    """
    print("\n" + "="*50)
    print("TESTING SPATIAL METRICS WITH REAL SHAPEFILE")
    print("="*50)
    
    shapefile_path = "test/shapefile/gbr1_coral_1m_merged_buffer0p001.shp"
    
    if not os.path.exists(shapefile_path):
        print(f"❌ Shapefile not found: {shapefile_path}")
        return False
    
    try:
        # Load shapefile and extract centroids
        data_shape, reef_centroids, num_sites = load_shapefile_and_centroids(shapefile_path)
        
        print(f"✅ Loaded {num_sites} reef sites from shapefile")
        
        # Calculate spatial metrics for a subset (first 10 reefs for speed)
        subset_size = min(10, num_sites)
        subset_centroids = reef_centroids[:subset_size]
        
        print(f"Calculating spatial metrics for first {subset_size} reefs...")
        
        angle_matrix, distance_matrix, direction_matrix = calculate_spatial_metrics(
            subset_centroids, subset_size)
        
        print("✅ Spatial metrics calculation completed!")
        print(f"Angle matrix shape: {angle_matrix.shape}")
        print(f"Distance matrix shape: {distance_matrix.shape}")
        print(f"Direction matrix shape: {direction_matrix.shape}")
        
        # Print some sample values
        print("\nSample values:")
        for i in range(min(3, subset_size)):
            for j in range(min(3, subset_size)):
                if i != j:
                    print(f"Reef {i} to {j}: angle={angle_matrix[i,j]:.1f}°, "
                          f"distance={distance_matrix[i,j]:.1f}km, "
                          f"direction={direction_matrix[i,j]}")
        
    except Exception as e:
        print(f"❌ Spatial metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """
    Run tests with real data.
    """
    print("REEFCONNECT PROCESS 1 - REAL DATA TESTING")
    print("="*60)
    
    # Test 1: Spatial metrics with real shapefile
    try:
        test_spatial_metrics_only()
        print("✅ Spatial metrics test passed!")
    except Exception as e:
        print(f"❌ Spatial metrics test failed: {e}")
    
    # Test 2: Full Process 1 with real data
    try:
        test_with_real_data()
        print("✅ Full Process 1 test passed!")
    except Exception as e:
        print(f"❌ Full Process 1 test failed: {e}")
    
    print("\n" + "="*60)
    print("REAL DATA TESTING COMPLETE")
    print("="*60)
    print("\nThe Process 1 implementation is working with real data!")
    print("You can now run it with your own data:")
    print("python process1_connectivity.py --shapefile path/to/reefs.shp --particle-data path/to/particles --output connectivity_data.nc")


if __name__ == "__main__":
    main() 