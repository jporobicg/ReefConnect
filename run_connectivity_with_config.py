#!/usr/bin/env python3
"""
Reef Connectivity Analysis with Configuration File
================================================

This script runs the connectivity analysis using parameters from the config file.
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from shapely.geometry import Point

# Import our modules
from io_utils import (
    load_config, load_shapefile_and_centroids, load_particle_data, load_sampled_trajectory_data,
    list_particle_files, extract_reef_id_from_filename, create_netcdf_output,
    verify_output_structure, filter_reefs_by_bounds, get_release_times_from_netcdf,
    calculate_species_day_weights, calculate_species_hour_weights, calculate_combined_day_hour_weights
)
from spatial_metrics import calculate_angles_and_distances
from ecological_processes import piecewise_decay, piecewise_competence, connolly_competence


def run_connectivity_analysis(config_path: str, shapefile_path: str = None, 
                            particle_data_path: str = None, output_path: str = None,
                            release_day: str = "2015-10-29"):
    """
    Run connectivity analysis using configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file.
    shapefile_path : str, optional
        Path to shapefile (overrides config).
    particle_data_path : str, optional
        Path to particle data directory (overrides config).
    output_path : str, optional
        Path for output file (overrides config).
    release_day : str
        Release day for particle data.
    """
    print("="*80)
    print("REEF CONNECTIVITY ANALYSIS WITH CONFIGURATION FILE")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Load configuration
    print(f"\n1. Loading configuration from: {config_path}")
    config = load_config(config_path)
    print(f"   ✅ Configuration loaded")
    print(f"   Bootstrap: {config['bootstrap']['sample_size']} particles, {config['bootstrap']['n_repetitions']} repetitions")
    
    # 2. Load shapefile and calculate spatial metrics
    print(f"\n2. Loading shapefile and calculating spatial metrics...")
    if shapefile_path is None:
        shapefile_path = config['default_paths']['shapefile']
    
    print(f"Loading shapefile: {shapefile_path}")
    data_shape, reef_centroids, num_sites = load_shapefile_and_centroids(shapefile_path)
    print(f"   ✅ Loaded {num_sites} reef sites")
    
    print(f"   Calculating spatial metrics...")
    angle_matrix, distance_matrix, direction_matrix = calculate_angles_and_distances(reef_centroids, num_sites)
    print(f"   ✅ Spatial metrics calculated: {angle_matrix.shape}")
    
    # 3. Find particle files
    print(f"\n3. Finding particle files for release day {release_day}...")
    if particle_data_path is None:
        particle_data_path = config['default_paths']['particle_data']
    
    particle_files = list_particle_files(particle_data_path, release_day)
    print(f"Found {len(particle_files)} particle files for release day {release_day}")
    print(f"   ✅ Found {len(particle_files)} particle files")
    
    # 4. Initialize connectivity matrix
    sample_size = config['bootstrap']['sample_size']
    n_repetitions = config['bootstrap']['n_repetitions']
    print(f"\n4. Initializing connectivity matrix...")
    print(f"   Dimensions: {num_sites} sources × {num_sites} sinks × {n_repetitions} samples")
    
    ## minimum age for competence
    tc = config['competence']['tc']
    t_min = config['connolly_competence']['t_min']
    cutt_off = min(tc, t_min)

    connectivity_matrix = np.zeros((num_sites, num_sites, 2, n_repetitions), dtype=np.float32)
    
    # 5. Process each particle file
    print(f"\n5. Processing particle files...")
    for i, particle_file in enumerate(particle_files):
        reef_id = extract_reef_id_from_filename(particle_file)
        print(f"   Processing reef {reef_id}...")       
     
        # Get trajectory information and release times
        output_nc = xr.open_dataset(particle_file)
        total_trajectories = output_nc.sizes['traj']
        output_nc.close()
        # Get species parameters from config
        species_config = config.get('species', {})
        species_name = species_config.get('name', 'acropora')
        use_day_weighting = species_config.get('use_day_weighting', True)
        use_hour_weighting = species_config.get('use_hour_weighting', True)
        use_combined_weighting = species_config.get('use_combined_weighting', True)
        
        # Get release times for weighting (if any weighting is enabled)
        if use_day_weighting or use_hour_weighting or use_combined_weighting:
            release_times, _ = get_release_times_from_netcdf(particle_file)
            print(f"     Release times: {len(release_times)} trajectories")
            print(f"     Weighting: Day={use_day_weighting}, Hour={use_hour_weighting}, Combined={use_combined_weighting}")
            
        
        for sample_idx in range(n_repetitions):
            # Sample trajectory indices using species-specific weighting
            if use_combined_weighting:
                # Use combined day and hour weighting
                weights = calculate_combined_day_hour_weights(release_times, species_name)
            elif use_day_weighting:
                # Use only day-based weights
                weights = calculate_species_day_weights(release_times, species_name)
            elif use_hour_weighting:
                # Use only hour-based weights
                weights = calculate_species_hour_weights(release_times, species_name)
            else:
                # Use simple random sampling
                weights = None
            
            # Sample with probability weights (if any)
            if weights is not None:
                if total_trajectories >= sample_size:
                    trajectory_indices = np.random.choice(total_trajectories, sample_size, replace=False, p=weights)
                else:
                    trajectory_indices = np.random.choice(total_trajectories, sample_size, replace=True, p=weights)
                
            else:
                # Use simple random sampling
                if total_trajectories >= sample_size:
                    trajectory_indices = np.random.choice(total_trajectories, sample_size, replace=False)
                else:
                    trajectory_indices = np.random.choice(total_trajectories, sample_size, replace=True)
            
            # Load only sampled trajectories directly from NetCDF
            sampled_particles, spatial_bounds = load_sampled_trajectory_data(particle_file, trajectory_indices, cutt_off)
            ## filter reefs based on particle spatial bounds
            ## this function returns the FID of the reefs that are within the particle spatial bounds
            candidate_reefs = filter_reefs_by_bounds(data_shape, spatial_bounds)
            # Calculate connectivity for this sample (only check candidate reefs)
            # Create a copy of particles for processing (to allow removal)
            particles_copy = sampled_particles.copy()
            
            for sink_reef_id in candidate_reefs:
                # Check if particles reached this sink reef
                sink_polygon = data_shape['geometry'][sink_reef_id]  # Use actual polygon, not centroid
                
                if len(particles_copy) == 0:
                    break  # No more particles to process
                
                # Find particles in this reef using vectorized approach
                particle_points = [Point(lon, lat) for lon, lat in zip(particles_copy['longitudes'], particles_copy['latitudes'])]
                in_polygon_mask = [sink_polygon.contains(point) for point in particle_points]
                settled_indices = np.where(in_polygon_mask)[0]
                
                if len(settled_indices) > 0:
                    # Get settled particles data
                    settled_particles = particles_copy.iloc[settled_indices]
                    settled_age = settled_particles['age'].values
                    settled_traj = settled_particles['trajectories'].values
                    
                    # Calculate decay and competence for all settled particles
                    decay = piecewise_decay(settled_age, 
                                          config['decay']['Tcp_decay'],
                                          config['decay']['lmbda1'], config['decay']['lmbda2'],
                                          config['decay']['v1'], config['decay']['v2'],
                                          config['decay']['sigma1'], config['decay']['sigma2'])
                    
                    # Competence probability (original piecewise model)
                    comp_prob = piecewise_competence(settled_age,
                                                   config['competence']['tc'], config['competence']['Tcp_comp'],
                                                   config['competence']['alpha'], config['competence']['beta1'],
                                                   config['competence']['beta2'], config['competence']['v'])
                    
                    # Competence probability (Connolly et al. 2010 model)
                    comp_prob_connolly = connolly_competence(settled_age,
                                                           config['connolly_competence']['t_min'], 
                                                           config['connolly_competence']['t_max'],
                                                           config['connolly_competence']['alpha'], 
                                                           config['connolly_competence']['beta'])
                    
                    # Create connectivity DataFrame for trajectory grouping
                    connect_df = pd.DataFrame({
                        'connect_moneghetti': np.array(decay) * np.array(comp_prob),
                        'connect_connolly': np.array(decay) * np.array(comp_prob_connolly),
                        'traj': settled_traj
                    })
                    
                    # Group by trajectory and calculate per-trajectory maximum
                    connect_max_moneghetti = (connect_df.groupby('traj')['connect_moneghetti'].max() / sample_size).values
                    connect_max_connolly = (connect_df.groupby('traj')['connect_connolly'].max() / sample_size).values
                    
                    # Average connectivity across trajectories
                    connectivity_moneghetti = connect_max_moneghetti.mean()
                    connectivity_connolly = connect_max_connolly.mean()
                    
                    # Store connectivity
                    connectivity_matrix[reef_id, sink_reef_id, 0, sample_idx] = connectivity_moneghetti
                    connectivity_matrix[reef_id, sink_reef_id, 1, sample_idx] = connectivity_connolly
                    
                    # Remove settled particles to avoid double-counting
                    particles_copy = particles_copy.drop(particles_copy.index[settled_indices]).reset_index(drop=True)
        
        print(f"     ✅ Completed {n_repetitions} bootstrap samples")
    
    print(f"\n   ✅ Processed {len(particle_files)}/{len(particle_files)} files successfully")
    
    # 6. Create output directory
    if output_path is None:
        output_dir = config['default_paths']['output_directory']
        output_path = os.path.join(output_dir, f"connectivity_results_{release_day}.nc")
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n6. Created output directory: {output_dir}")
    
    # 7. Save results to NetCDF
    print(f"\n7. Saving results to NetCDF...")
    print(f"   Output file: {output_path}")
    
    create_netcdf_output(
        output_path, num_sites, num_sites, n_repetitions,
        angle_matrix, distance_matrix, direction_matrix, connectivity_matrix
    )
    
    # 8. Verify output
    print(f"\n8. Verifying output...")
    verification_ok = verify_output_structure(output_path, num_sites, num_sites, n_repetitions)
    
    if verification_ok:
        file_size = os.path.getsize(output_path)
        print(f"   ✅ Output verified successfully")
        print(f"   File size: {file_size:,} bytes")
    else:
        print(f"   ❌ Output verification failed")
        return False
    
    # 9. Summary
    processing_time = time.time() - start_time
    print(f"\n" + "="*80)
    print(f"ANALYSIS COMPLETE")
    print(f"="*80)
    print(f"✅ Processing time: {processing_time:.2f} seconds")
    print(f"✅ Input files: {len(particle_files)} particle files")
    print(f"✅ Processed files: {len(particle_files)}")
    print(f"✅ Output: {output_path}")
    print(f"✅ Matrix dimensions: {num_sites}×{num_sites}×2 treatments×{n_repetitions} samples")
    
    return True


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Run reef connectivity analysis using configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_connectivity_with_config.py
  python run_connectivity_with_config.py --config config/connectivity_parameters.yaml
  python run_connectivity_with_config.py --shapefile test/shapefile/gbr1_coral_1m_merged_buffer0p001.shp --particle-data test/2015-10-29/ --output output/my_results.nc
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/connectivity_parameters.yaml',
        help='Path to configuration file (default: config/connectivity_parameters.yaml)'
    )
    
    parser.add_argument(
        '--shapefile', '-s',
        help='Path to shapefile (overrides config)'
    )
    
    parser.add_argument(
        '--particle-data', '-p',
        help='Path to particle data directory (overrides config)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Path for output file (overrides config)'
    )
    
    parser.add_argument(
        '--release-day', '-r',
        default='2015-10-29',
        help='Release day for particle data (default: 2015-10-29)'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Run analysis
    success = run_connectivity_analysis(
        config_path=args.config,
        shapefile_path=args.shapefile,
        particle_data_path=args.particle_data,
        output_path=args.output,
        release_day=args.release_day
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main() 