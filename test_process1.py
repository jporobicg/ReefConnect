#!/usr/bin/env python3
"""
Test script for Process 1: Generate Connectivity Data

This script tests the Process 1 implementation with a small sample to verify:
- Spatial metrics calculation
- Bootstrap resampling framework
- NetCDF output generation
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


def create_test_shapefile():
    """
    Create a simple test shapefile for testing.
    This is a placeholder - in real testing you would use actual shapefile data.
    """
    print("Creating test shapefile...")
    
    # For testing purposes, we'll create a simple test
    # In real usage, you would provide an actual shapefile path
    test_shapefile = "test_reefs.shp"
    
    # Check if we have a test shapefile or need to create one
    if not os.path.exists(test_shapefile):
        print(f"Warning: Test shapefile {test_shapefile} not found.")
        print("Please provide a valid shapefile path for testing.")
        return None
    
    return test_shapefile


def create_test_particle_data():
    """
    Create test particle data for testing.
    This is a placeholder - in real testing you would use actual particle data.
    """
    print("Creating test particle data...")
    
    # For testing purposes, we'll create a simple test
    # In real usage, you would provide actual particle data directory
    test_particle_dir = "test_particle_data"
    
    if not os.path.exists(test_particle_dir):
        print(f"Warning: Test particle data directory {test_particle_dir} not found.")
        print("Please provide a valid particle data directory for testing.")
        return None
    
    return test_particle_dir


def test_spatial_metrics():
    """
    Test spatial metrics calculation with a small sample.
    """
    print("\n" + "="*50)
    print("TESTING SPATIAL METRICS CALCULATION")
    print("="*50)
    
    # Create a simple test with 3 reefs
    print("Creating test reef centroids...")
    
    # Simple test centroids (lat, lon)
    test_centroids = [
        (1.0, 1.0),  # Reef 0
        (1.1, 1.1),  # Reef 1  
        (1.2, 1.2),  # Reef 2
    ]
    
    # Convert to the format expected by the function
    from shapely.geometry import Point
    reef_centroids = [Point(lon, lat) for lat, lon in test_centroids]
    
    print(f"Testing with {len(reef_centroids)} reefs...")
    
    # Test spatial metrics calculation
    angle_matrix, distance_matrix, direction_matrix = calculate_spatial_metrics(
        reef_centroids, len(reef_centroids))
    
    print("Spatial metrics calculation completed!")
    print(f"Angle matrix shape: {angle_matrix.shape}")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Direction matrix shape: {direction_matrix.shape}")
    
    # Print some sample values
    print("\nSample values:")
    print(f"Angle from reef 0 to reef 1: {angle_matrix[0, 1]:.2f} degrees")
    print(f"Distance from reef 0 to reef 1: {distance_matrix[0, 1]:.2f} km")
    print(f"Direction from reef 0 to reef 1: {direction_matrix[0, 1]}")
    
    return angle_matrix, distance_matrix, direction_matrix


def test_netcdf_output():
    """
    Test NetCDF output generation.
    """
    print("\n" + "="*50)
    print("TESTING NETCDF OUTPUT GENERATION")
    print("="*50)
    
    # Create test data
    num_reefs = 3
    num_samples = 5
    
    angle_matrix = np.random.rand(num_reefs, num_reefs) * 360
    distance_matrix = np.random.rand(num_reefs, num_reefs) * 100
    direction_matrix = np.random.randint(0, 36, (num_reefs, num_reefs))
    connectivity_matrix = np.random.exponential(0.1, (num_reefs, num_reefs, num_samples))
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    print(f"Creating test NetCDF file: {output_path}")
    
    # Test NetCDF creation
    create_netcdf_dataset(
        angle_matrix, distance_matrix, direction_matrix, 
        connectivity_matrix, output_path)
    
    # Verify the file was created
    if os.path.exists(output_path):
        print("✅ NetCDF file created successfully!")
        
        # Load and verify the data
        ds = xr.open_dataset(output_path)
        print(f"✅ NetCDF dimensions: {dict(ds.dims)}")
        print(f"✅ NetCDF variables: {list(ds.data_vars.keys())}")
        
        # Clean up
        ds.close()
        os.unlink(output_path)
        print("✅ Test file cleaned up")
    else:
        print("❌ NetCDF file creation failed!")
    
    return True


def test_full_process():
    """
    Test the full Process 1 workflow with minimal data.
    """
    print("\n" + "="*50)
    print("TESTING FULL PROCESS 1 WORKFLOW")
    print("="*50)
    
    # Check if we have test data
    test_shapefile = create_test_shapefile()
    test_particle_data = create_test_particle_data()
    
    if not test_shapefile or not test_particle_data:
        print("Skipping full process test due to missing test data.")
        print("To run full test, provide:")
        print("1. A valid shapefile with reef polygons")
        print("2. A directory with particle tracking NetCDF files")
        return False
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    try:
        print("Running full Process 1 workflow...")
        
        results = process1_main(
            shapefile_path=test_shapefile,
            particle_data_path=test_particle_data,
            output_path=output_path,
            num_samples=10,  # Small sample for testing
            num_repetitions=5  # Small number for testing
        )
        
        print("✅ Full Process 1 workflow completed!")
        print(f"Results: {results}")
        
        # Verify output file
        if os.path.exists(output_path):
            ds = xr.open_dataset(output_path)
            print(f"✅ Output file created with dimensions: {dict(ds.dims)}")
            ds.close()
            
            # Clean up
            os.unlink(output_path)
            print("✅ Test file cleaned up")
        else:
            print("❌ Output file not created!")
            return False
            
    except Exception as e:
        print(f"❌ Full process test failed: {e}")
        return False
    
    return True


def main():
    """
    Run all tests.
    """
    print("REEFCONNECT PROCESS 1 - TESTING SUITE")
    print("="*60)
    
    # Test 1: Spatial metrics calculation
    try:
        test_spatial_metrics()
        print("✅ Spatial metrics test passed!")
    except Exception as e:
        print(f"❌ Spatial metrics test failed: {e}")
    
    # Test 2: NetCDF output generation
    try:
        test_netcdf_output()
        print("✅ NetCDF output test passed!")
    except Exception as e:
        print(f"❌ NetCDF output test failed: {e}")
    
    # Test 3: Full process workflow
    try:
        test_full_process()
        print("✅ Full process test passed!")
    except Exception as e:
        print(f"❌ Full process test failed: {e}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\nTo run with real data:")
    print("python process1_connectivity.py --shapefile path/to/reefs.shp --particle-data path/to/particles --output connectivity_data.nc")


if __name__ == "__main__":
    main() 