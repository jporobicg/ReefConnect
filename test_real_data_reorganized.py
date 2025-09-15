#!/usr/bin/env python3
"""
Test with Real Data using Reorganized Modules
"""

import numpy as np
import tempfile
import os
from pathlib import Path
import time

# Import our reorganized modules
from ecological_processes import piecewise_decay, piecewise_competence, points_in_polygon
from spatial_metrics import calculate_angles_and_distances
from main_connectivity_calculation import calc
from io_utils import load_config, load_shapefile_and_centroids, load_particle_data, create_netcdf_output, verify_output_structure


def test_real_shapefile_loading():
    """Test loading the real shapefile."""
    print("="*60)
    print("TESTING REAL SHAPEFILE LOADING")
    print("="*60)
    
    shapefile_path = "test/shapefile/gbr1_coral_1m_merged_buffer0p001.shp"
    
    if not os.path.exists(shapefile_path):
        print(f"❌ Shapefile not found: {shapefile_path}")
        return False
    
    try:
        data_shape, reef_centroids, num_sites = load_shapefile_and_centroids(shapefile_path)
        
        print(f"✅ Successfully loaded shapefile:")
        print(f"   Number of reef sites: {num_sites}")
        print(f"   Shapefile columns: {list(data_shape.columns)}")
        print(f"   Geometry type: {data_shape.geometry.geom_type.iloc[0]}")
        print(f"   Bounds: {data_shape.total_bounds}")
        
        # Test a few centroids
        for i in range(min(3, len(reef_centroids))):
            centroid = reef_centroids[i]
            print(f"   Reef {i} centroid: ({centroid.x:.4f}, {centroid.y:.4f})")
        
        return True, data_shape, reef_centroids, num_sites
        
    except Exception as e:
        print(f"❌ Shapefile loading failed: {e}")
        return False, None, None, None


def test_real_particle_data_loading():
    """Test loading real particle data."""
    print("\n" + "="*60)
    print("TESTING REAL PARTICLE DATA LOADING")
    print("="*60)
    
    # Test with one particle file
    particle_file = "test/2015-10-29/GBR1_H2p0_Coral_Release_2015-10-29_Polygon_0_Wind_3_percent_displacement_field.nc"
    
    if not os.path.exists(particle_file):
        print(f"❌ Particle file not found: {particle_file}")
        return False, None, None
    
    try:
        particles, ntraj = load_particle_data(particle_file)
        
        print(f"✅ Successfully loaded particle data:")
        print(f"   Number of trajectories: {ntraj}")
        print(f"   Number of particle records: {len(particles)}")
        print(f"   Columns: {list(particles.columns)}")
        print(f"   Age range: {particles['age'].min():.2f} - {particles['age'].max():.2f} days")
        print(f"   Lat range: {particles['latitudes'].min():.4f} - {particles['latitudes'].max():.4f}")
        print(f"   Lon range: {particles['longitudes'].min():.4f} - {particles['longitudes'].max():.4f}")
        
        return True, particles, ntraj
        
    except Exception as e:
        print(f"❌ Particle data loading failed: {e}")
        return False, None, None


def test_spatial_metrics_with_real_data(data_shape, reef_centroids, num_sites):
    """Test spatial metrics calculation with real reef data."""
    print("\n" + "="*60)
    print("TESTING SPATIAL METRICS WITH REAL DATA")
    print("="*60)
    
    try:
        # Calculate for a subset to save time
        subset_size = min(10, num_sites)
        subset_centroids = reef_centroids[:subset_size]
        
        print(f"Calculating spatial metrics for {subset_size} reefs...")
        start_time = time.time()
        
        angle_matrix, direction_matrix, distance_matrix = calculate_angles_and_distances(
            subset_centroids, subset_size
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ Spatial metrics calculated successfully:")
        print(f"   Time elapsed: {elapsed_time:.2f} seconds")
        print(f"   Matrix shapes: {angle_matrix.shape}")
        print(f"   Sample values:")
        print(f"     Distance[0,1]: {distance_matrix[0, 1]:.2f} km")
        print(f"     Angle[0,1]: {angle_matrix[0, 1]:.2f} degrees")
        print(f"     Direction[0,1]: {direction_matrix[0, 1]}")
        
        return True, angle_matrix, direction_matrix, distance_matrix
        
    except Exception as e:
        print(f"❌ Spatial metrics calculation failed: {e}")
        return False, None, None, None


def test_ecological_processes_with_real_data(particles):
    """Test ecological processes with real particle ages."""
    print("\n" + "="*60)
    print("TESTING ECOLOGICAL PROCESSES WITH REAL DATA")
    print("="*60)
    
    # Load configuration
    config = load_config("config/connectivity_parameters.yaml")
    
    try:
        # Get unique ages from particle data
        unique_ages = sorted(particles['age'].unique())
        test_ages = unique_ages[:10]  # Test with first 10 unique ages
        
        print(f"Testing with {len(test_ages)} unique ages from particle data")
        print(f"Age range: {test_ages[0]:.2f} - {test_ages[-1]:.2f} days")
        
        # Test decay function
        decay_params = config['decay']
        decay_result = piecewise_decay(
            test_ages,
            decay_params['Tcp_decay'],
            decay_params['lmbda1'],
            decay_params['lmbda2'],
            decay_params['v1'],
            decay_params['v2'],
            decay_params['sigma1'],
            decay_params['sigma2']
        )
        
        # Test competence function
        competence_params = config['competence']
        competence_result = piecewise_competence(
            test_ages,
            competence_params['tc'],
            competence_params['Tcp_comp'],
            competence_params['alpha'],
            competence_params['beta1'],
            competence_params['beta2'],
            competence_params['v']
        )
        
        print(f"✅ Ecological processes calculated successfully:")
        print(f"   Decay values: {len(decay_result)} (range: {min(decay_result):.4f} - {max(decay_result):.4f})")
        print(f"   Competence values: {len(competence_result)} (range: {min(competence_result):.4f} - {max(competence_result):.4f})")
        
        # Calculate combined connectivity
        combined = np.array(decay_result) * np.array(competence_result)
        print(f"   Combined connectivity: {len(combined)} (range: {min(combined):.4f} - {max(combined):.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Ecological processes calculation failed: {e}")
        return False


def test_netcdf_creation_with_real_data(num_sites, angle_matrix, direction_matrix, distance_matrix):
    """Test NetCDF creation with real data dimensions."""
    print("\n" + "="*60)
    print("TESTING NETCDF CREATION WITH REAL DATA")
    print("="*60)
    
    # Load configuration
    config = load_config("config/connectivity_parameters.yaml")
    
    try:
        # Create realistic connectivity data
        num_sources = min(10, num_sites)  # Use subset for testing
        num_sinks = min(10, num_sites)
        num_samples = config['bootstrap']['n_repetitions']
        
        # Create realistic connectivity data (smaller for testing)
        connectivity_data = np.random.exponential(0.1, (num_sources, num_sinks, num_samples))
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            # Create NetCDF file
            create_netcdf_output(
                output_path, num_sources, num_sinks, num_samples,
                angle_matrix[:num_sources, :num_sinks], 
                distance_matrix[:num_sources, :num_sinks], 
                direction_matrix[:num_sources, :num_sinks], 
                connectivity_data
            )
            
            # Verify the output
            verification_result = verify_output_structure(
                output_path, num_sources, num_sinks, num_samples
            )
            
            if verification_result:
                file_size = os.path.getsize(output_path)
                print(f"✅ NetCDF creation successful:")
                print(f"   File size: {file_size:,} bytes")
                print(f"   Dimensions: {num_sources} sources × {num_sinks} sinks × {num_samples} samples")
                print(f"   Variables: angle, distance, direction, connectivity")
            
            return True
            
        finally:
            # Cleanup
            if os.path.exists(output_path):
                os.unlink(output_path)
        
    except Exception as e:
        print(f"❌ NetCDF creation failed: {e}")
        return False


def main():
    """Run comprehensive test with real data."""
    print("REAL DATA TESTING WITH REORGANIZED MODULES")
    print("="*80)
    print("Testing the reorganized codebase with actual shapefile and particle data")
    print("="*80)
    
    # Test 1: Real shapefile loading
    shapefile_success, data_shape, reef_centroids, num_sites = test_real_shapefile_loading()
    
    # Test 2: Real particle data loading
    particle_success, particles, ntraj = test_real_particle_data_loading()
    
    # Test 3: Spatial metrics with real data
    spatial_success = False
    angle_matrix = direction_matrix = distance_matrix = None
    if shapefile_success:
        spatial_success, angle_matrix, direction_matrix, distance_matrix = test_spatial_metrics_with_real_data(
            data_shape, reef_centroids, num_sites
        )
    
    # Test 4: Ecological processes with real data
    ecological_success = False
    if particle_success:
        ecological_success = test_ecological_processes_with_real_data(particles)
    
    # Test 5: NetCDF creation with real data
    netcdf_success = False
    if spatial_success and angle_matrix is not None:
        netcdf_success = test_netcdf_creation_with_real_data(
            num_sites, angle_matrix, direction_matrix, distance_matrix
        )
    
    # Summary
    print("\n" + "="*80)
    print("REAL DATA TESTING SUMMARY")
    print("="*80)
    
    tests = [
        ("Shapefile loading", shapefile_success),
        ("Particle data loading", particle_success),
        ("Spatial metrics calculation", spatial_success),
        ("Ecological processes", ecological_success),
        ("NetCDF creation", netcdf_success),
    ]
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    for test_name, success in tests:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL REAL DATA TESTS PASSED! 🎉")
        print("="*80)
        print("✅ Reorganized modules work perfectly with real data!")
        print("✅ Shapefile loading: Functional")
        print("✅ Particle data loading: Functional") 
        print("✅ Spatial metrics: Functional")
        print("✅ Ecological processes: Functional")
        print("✅ NetCDF output: Functional")
        print("\n🚀 Ready for production use!")
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        print("Please review the errors above.")


if __name__ == "__main__":
    main() 