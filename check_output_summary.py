#!/usr/bin/env python3
"""
Check Output Summary
===================

Quick script to show a summary of the generated NetCDF output file.
"""

import xarray as xr
import numpy as np

def check_output_summary(file_path):
    """Check and display summary of NetCDF output file."""
    print("="*60)
    print("OUTPUT NETCDF SUMMARY")
    print("="*60)
    
    # Open the dataset
    ds = xr.open_dataset(file_path)
    
    print(f"File: {file_path}")
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Variables: {list(ds.variables.keys())}")
    print(f"File size: {ds.nbytes / 1e9:.2f} GB")
    print()
    
    # Show sample data for each variable
    print("SAMPLE DATA:")
    print("-" * 40)
    
    # Angle data
    angle_data = ds['angle'].values
    print(f"Angle matrix shape: {angle_data.shape}")
    print(f"Angle range: {np.min(angle_data):.2f} - {np.max(angle_data):.2f} degrees")
    print(f"Sample angles (0,1): {angle_data[0, 1]:.2f}°")
    print(f"Sample angles (1,2): {angle_data[1, 2]:.2f}°")
    print()
    
    # Distance data
    distance_data = ds['distance'].values
    print(f"Distance matrix shape: {distance_data.shape}")
    print(f"Distance range: {np.min(distance_data):.2f} - {np.max(distance_data):.2f} km")
    print(f"Sample distance (0,1): {distance_data[0, 1]:.2f} km")
    print(f"Sample distance (1,2): {distance_data[1, 2]:.2f} km")
    print()
    
    # Direction data
    direction_data = ds['direction'].values
    print(f"Direction matrix shape: {direction_data.shape}")
    print(f"Direction range: {np.min(direction_data):.0f} - {np.max(direction_data):.0f} sectors")
    print(f"Sample direction (0,1): {direction_data[0, 1]:.0f}")
    print(f"Sample direction (1,2): {direction_data[1, 2]:.0f}")
    print()
    
    # Connectivity data
    connectivity_data = ds['connectivity'].values
    print(f"Connectivity matrix shape: {connectivity_data.shape}")
    print(f"Connectivity range: {np.min(connectivity_data):.6f} - {np.max(connectivity_data):.6f}")
    print(f"Sample connectivity (0,0,0): {connectivity_data[0, 0, 0]:.6f}")
    print(f"Sample connectivity (0,1,0): {connectivity_data[0, 1, 0]:.6f}")
    print(f"Sample connectivity (1,2,0): {connectivity_data[1, 2, 0]:.6f}")
    print()
    
    # Show some statistics
    print("STATISTICS:")
    print("-" * 40)
    print(f"Non-zero connectivity values: {np.count_nonzero(connectivity_data)}")
    print(f"Connectivity mean: {np.mean(connectivity_data):.6f}")
    print(f"Connectivity std: {np.std(connectivity_data):.6f}")
    print()
    
    # Check for any issues
    print("VALIDATION:")
    print("-" * 40)
    
    # Check for NaN values
    nan_count = np.isnan(connectivity_data).sum()
    print(f"NaN values in connectivity: {nan_count}")
    
    # Check for infinite values
    inf_count = np.isinf(connectivity_data).sum()
    print(f"Infinite values in connectivity: {inf_count}")
    
    # Check diagonal elements (self-connectivity)
    diagonal_values = np.diagonal(connectivity_data, axis1=0, axis2=1)
    print(f"Self-connectivity mean: {np.mean(diagonal_values):.6f}")
    print(f"Self-connectivity max: {np.max(diagonal_values):.6f}")
    
    ds.close()
    
    print("\n" + "="*60)
    print("SUMMARY COMPLETE")
    print("="*60)


if __name__ == "__main__":
    check_output_summary("output/connectivity_results_example.nc") 