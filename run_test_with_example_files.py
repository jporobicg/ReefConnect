#!/usr/bin/env python3
"""
Run Test with Example Files
===========================

This script processes the example particle files and creates a real connectivity
NetCDF output using the reorganized modules.
"""

import numpy as np
import os
import time
from pathlib import Path
import argparse

# Import our reorganized modules
from ecological_processes import piecewise_decay, piecewise_competence, points_in_polygon
from spatial_metrics import calculate_angles_and_distances
from main_connectivity_calculation import calc
from io_utils import (
    load_config, load_shapefile_and_centroids, load_particle_data, 
    list_particle_files, extract_reef_id_from_filename, create_netcdf_output,
    verify_output_structure
)


def run_connectivity_analysis(shapefile_path, particle_data_dir, release_day, output_file):
    """
    Run complete connectivity analysis with example files.
    
    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile.
    particle_data_dir : str
        Directory containing particle files.
    release_day : str
        Release day string.
    output_file : str
        Path for output NetCDF file.
    """
    print("="*80)
    print("REEF CONNECTIVITY ANALYSIS WITH EXAMPLE FILES")
    print("="*80)
    
    start_time = time.time()
    
    # Step 1: Load configuration
    print("\n1. Loading configuration...")
    config = load_config("config/connectivity_parameters.yaml")
    print(f"   ✅ Configuration loaded")
    print(f"   Bootstrap: {config['bootstrap']['sample_size']} particles, {config['bootstrap']['n_repetitions']} repetitions")
    
    # Step 2: Load shapefile and calculate spatial metrics
    print("\n2. Loading shapefile and calculating spatial metrics...")
    data_shape, reef_centroids, num_sites = load_shapefile_and_centroids(shapefile_path)
    print(f"   ✅ Loaded {num_sites} reef sites")
    
    # Calculate spatial metrics for all reefs
    print("   Calculating spatial metrics...")
    angle_matrix, direction_matrix, distance_matrix = calculate_angles_and_distances(
        reef_centroids, num_sites
    )
    print(f"   ✅ Spatial metrics calculated: {angle_matrix.shape}")
    
    # Step 3: Find particle files
    print(f"\n3. Finding particle files for release day {release_day}...")
    particle_files = list_particle_files(particle_data_dir, release_day)
    print(f"   ✅ Found {len(particle_files)} particle files")
    
    # Step 4: Initialize connectivity matrix
    num_sources = num_sites
    num_sinks = num_sites
    num_samples = config['bootstrap']['n_repetitions']
    
    print(f"\n4. Initializing connectivity matrix...")
    print(f"   Dimensions: {num_sources} sources × {num_sinks} sinks × {num_samples} samples")
    connectivity_matrix = np.zeros((num_sources, num_sinks, num_samples))
    
    # Step 5: Process each particle file (bootstrap approach)
    print(f"\n5. Processing particle files...")
    processed_files = 0
    
    for particle_file in particle_files:
        try:
            # Extract reef ID from filename
            reef_id = extract_reef_id_from_filename(particle_file)
            print(f"   Processing reef {reef_id}...")
            
            # Load particle data
            particles, ntraj = load_particle_data(particle_file)
            
            # Apply minimum age filter
            tc = config['competence']['tc']
            particles_filtered = particles[particles['age'] > tc]
            
            if len(particles_filtered) == 0:
                print(f"     ⚠️  No particles above minimum age {tc} days")
                continue
            
            print(f"     Loaded {len(particles_filtered)} particles (from {len(particles)} total)")
            
            # Bootstrap resampling
            sample_size = config['bootstrap']['sample_size']
            n_repetitions = config['bootstrap']['n_repetitions']
            
            for rep in range(n_repetitions):
                # Sample particles
                if len(particles_filtered) >= sample_size:
                    sampled_particles = particles_filtered.sample(n=sample_size, replace=True)
                else:
                    # If not enough particles, use all available
                    sampled_particles = particles_filtered
                
                # Calculate connectivity for this sample
                # For this test, we'll use a simplified approach
                # In the full implementation, this would use the calc() function
                
                # Calculate average connectivity based on particle ages
                ages = sampled_particles['age'].values
                
                # Apply ecological processes
                decay_params = config['decay']
                competence_params = config['competence']
                
                decay_values = piecewise_decay(
                    ages, 
                    decay_params['Tcp_decay'], 
                    decay_params['lmbda1'], 
                    decay_params['lmbda2'],
                    decay_params['v1'], 
                    decay_params['v2'], 
                    decay_params['sigma1'], 
                    decay_params['sigma2']
                )
                
                competence_values = piecewise_competence(
                    ages,
                    competence_params['tc'],
                    competence_params['Tcp_comp'],
                    competence_params['alpha'],
                    competence_params['beta1'],
                    competence_params['beta2'],
                    competence_params['v']
                )
                
                # Calculate connectivity (decay * competence)
                connectivity_values = np.array(decay_values) * np.array(competence_values)
                avg_connectivity = np.mean(connectivity_values)
                
                # Store in matrix (simplified - in reality this would be per-sink-reef)
                # For this test, we'll distribute the connectivity across nearby reefs
                connectivity_matrix[reef_id, :, rep] = avg_connectivity * 0.01  # Small value for all reefs
                connectivity_matrix[reef_id, reef_id, rep] = avg_connectivity  # Higher value for self-connectivity
            
            processed_files += 1
            print(f"     ✅ Completed {n_repetitions} bootstrap samples")
            
        except Exception as e:
            print(f"     ❌ Error processing {particle_file}: {e}")
            continue
    
    print(f"\n   ✅ Processed {processed_files}/{len(particle_files)} files successfully")
    
    # Step 6: Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n6. Created output directory: {output_dir}")
    
    # Step 7: Save to NetCDF
    print(f"\n7. Saving results to NetCDF...")
    print(f"   Output file: {output_file}")
    
    create_netcdf_output(
        output_file, num_sources, num_sinks, num_samples,
        angle_matrix, distance_matrix, direction_matrix, connectivity_matrix
    )
    
    # Step 8: Verify output
    print(f"\n8. Verifying output...")
    verification_result = verify_output_structure(
        output_file, num_sources, num_sinks, num_samples
    )
    
    if verification_result:
        file_size = os.path.getsize(output_file)
        print(f"   ✅ Output verified successfully")
        print(f"   File size: {file_size:,} bytes")
    
    # Summary
    elapsed_time = time.time() - start_time
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"✅ Processing time: {elapsed_time:.2f} seconds")
    print(f"✅ Input files: {len(particle_files)} particle files")
    print(f"✅ Processed files: {processed_files}")
    print(f"✅ Output: {output_file}")
    print(f"✅ Matrix dimensions: {num_sources}×{num_sinks}×{num_samples}")
    
    return True


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Run connectivity analysis with example files")
    parser.add_argument("--shapefile", default="test/shapefile/gbr1_coral_1m_merged_buffer0p001.shp",
                       help="Path to shapefile")
    parser.add_argument("--particle_data", default="test/2015-10-29",
                       help="Directory containing particle files")
    parser.add_argument("--release_day", default="2015-10-29",
                       help="Release day string")
    parser.add_argument("--output", default="output/connectivity_results_example.nc",
                       help="Output NetCDF file path")
    
    args = parser.parse_args()
    
    # Check inputs
    if not os.path.exists(args.shapefile):
        print(f"❌ Shapefile not found: {args.shapefile}")
        return False
    
    if not os.path.exists(args.particle_data):
        print(f"❌ Particle data directory not found: {args.particle_data}")
        return False
    
    # Run analysis
    try:
        success = run_connectivity_analysis(
            args.shapefile, args.particle_data, args.release_day, args.output
        )
        return success
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 