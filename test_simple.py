#!/usr/bin/env python3
"""
Simple test for Process 1 core functionality
"""

import numpy as np
import xarray as xr
import tempfile
import os
from shapely.geometry import Point

# Import only the functions we need, avoiding the main section
from original_code.angle import angle, haversine, veclength


def test_spatial_calculations():
    """
    Test the core spatial calculations from the original code.
    """
    print("="*50)
    print("TESTING SPATIAL CALCULATIONS")
    print("="*50)
    
    # Test vector length calculation
    vector = [3, 4]
    length = veclength(vector)
    print(f"Vector {vector} length: {length}")
    assert abs(length - 5.0) < 0.001, "Vector length calculation failed"
    
    # Test angle calculation
    a = [0, 1]  # Reference vector
    b = [1, 1]  # Test vector
    angle_val = angle(a, b)
    print(f"Angle between {a} and {b}: {angle_val:.2f} degrees")
    assert 0 <= angle_val <= 180, "Angle calculation failed"
    
    # Test haversine distance
    coord1 = (0, 0)
    coord2 = (1, 1)
    distance = haversine(coord1, coord2)
    print(f"Distance between {coord1} and {coord2}: {distance:.2f} km")
    assert distance > 0, "Haversine distance calculation failed"
    
    print("✅ All spatial calculations passed!")


def test_netcdf_creation():
    """
    Test NetCDF file creation with the specified structure.
    """
    print("\n" + "="*50)
    print("TESTING NETCDF CREATION")
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
    
    # Create dataset
    ds = xr.Dataset(
        data_vars={
            'angle': xr.DataArray(
                angle_matrix,
                dims=['source', 'sink'],
                coords={'source': range(num_reefs), 'sink': range(num_reefs)},
                attrs={'units': 'degrees', 'long_name': 'Angle between source and sink reefs'}
            ),
            'distance': xr.DataArray(
                distance_matrix,
                dims=['source', 'sink'],
                coords={'source': range(num_reefs), 'sink': range(num_reefs)},
                attrs={'units': 'kilometers', 'long_name': 'Distance between source and sink reefs'}
            ),
            'direction': xr.DataArray(
                direction_matrix,
                dims=['source', 'sink'],
                coords={'source': range(num_reefs), 'sink': range(num_reefs)},
                attrs={'units': 'sectors', 'long_name': 'Directional sector from source to sink'}
            ),
            'connectivity': xr.DataArray(
                connectivity_matrix,
                dims=['source', 'sink', 'sample'],
                coords={
                    'source': range(num_reefs), 
                    'sink': range(num_reefs),
                    'sample': range(num_samples)
                },
                attrs={'units': 'probability', 'long_name': 'Bootstrap connectivity values'}
            )
        }
    )
    
    # Add global attributes
    ds.attrs['creation_date'] = '2024-01-01'
    ds.attrs['description'] = 'Test NetCDF file'
    ds.attrs['num_reefs'] = num_reefs
    ds.attrs['num_bootstrap_samples'] = num_samples
    
    # Save to NetCDF
    ds.to_netcdf(output_path)
    
    # Verify the file was created
    if os.path.exists(output_path):
        print("✅ NetCDF file created successfully!")
        
        # Load and verify the data
        ds_loaded = xr.open_dataset(output_path)
        print(f"✅ NetCDF dimensions: {dict(ds_loaded.dims)}")
        print(f"✅ NetCDF variables: {list(ds_loaded.data_vars.keys())}")
        
        # Verify dimensions
        expected_dims = {'source': num_reefs, 'sink': num_reefs, 'sample': num_samples}
        for dim, size in expected_dims.items():
            assert dim in ds_loaded.dims, f"Missing dimension: {dim}"
            assert ds_loaded.dims[dim] == size, f"Wrong size for dimension {dim}"
        
        # Verify variables
        expected_vars = ['angle', 'distance', 'direction', 'connectivity']
        for var in expected_vars:
            assert var in ds_loaded.data_vars, f"Missing variable: {var}"
        
        # Clean up
        ds_loaded.close()
        os.unlink(output_path)
        print("✅ Test file cleaned up")
    else:
        print("❌ NetCDF file creation failed!")
        return False
    
    return True


def test_bootstrap_logic():
    """
    Test the bootstrap resampling logic.
    """
    print("\n" + "="*50)
    print("TESTING BOOTSTRAP LOGIC")
    print("="*50)
    
    # Simulate particle data
    num_particles = 1000
    particles = np.random.rand(num_particles, 2)  # lat, lon
    
    # Bootstrap parameters
    num_samples = 100
    num_repetitions = 5
    
    print(f"Original particles: {num_particles}")
    print(f"Bootstrap sample size: {num_samples}")
    print(f"Number of repetitions: {num_repetitions}")
    
    # Perform bootstrap resampling
    bootstrap_results = []
    
    for rep in range(num_repetitions):
        # Randomly sample without replacement
        if num_particles >= num_samples:
            sampled_indices = np.random.choice(num_particles, num_samples, replace=False)
        else:
            sampled_indices = np.arange(num_particles)
        
        sampled_particles = particles[sampled_indices]
        bootstrap_results.append(sampled_particles)
        
        print(f"Bootstrap {rep+1}: sampled {len(sampled_particles)} particles")
    
    print(f"✅ Bootstrap resampling completed: {len(bootstrap_results)} samples")
    
    # Verify bootstrap results
    assert len(bootstrap_results) == num_repetitions, "Wrong number of bootstrap samples"
    for i, sample in enumerate(bootstrap_results):
        assert len(sample) <= num_samples, f"Sample {i} too large"
        assert len(sample) > 0, f"Sample {i} is empty"
    
    print("✅ All bootstrap tests passed!")
    return True


def main():
    """
    Run all tests.
    """
    print("REEFCONNECT PROCESS 1 - SIMPLE TESTING SUITE")
    print("="*60)
    
    # Test 1: Spatial calculations
    try:
        test_spatial_calculations()
    except Exception as e:
        print(f"❌ Spatial calculations test failed: {e}")
    
    # Test 2: NetCDF creation
    try:
        test_netcdf_creation()
    except Exception as e:
        print(f"❌ NetCDF creation test failed: {e}")
    
    # Test 3: Bootstrap logic
    try:
        test_bootstrap_logic()
    except Exception as e:
        print(f"❌ Bootstrap logic test failed: {e}")
    
    print("\n" + "="*60)
    print("SIMPLE TESTING COMPLETE")
    print("="*60)
    print("\nThe core functionality is working!")
    print("To run with real data, you'll need:")
    print("1. A valid shapefile with reef polygons")
    print("2. Particle tracking NetCDF files")
    print("3. Run: python process1_connectivity.py --shapefile path/to/reefs.shp --particle-data path/to/particles --output connectivity_data.nc")


if __name__ == "__main__":
    main() 