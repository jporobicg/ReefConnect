#!/usr/bin/env python3
"""
Reef Connectivity Analysis with Parallel Processing
=================================================

This script runs the connectivity analysis using parallel processing with joblib.
"""

import argparse
import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from shapely.geometry import Point
from joblib import Parallel, delayed, parallel_backend

# Import our modules
from io_utils import (
    load_config, load_shapefile_and_centroids, load_particle_data, load_sampled_trajectory_data,
    list_particle_files, extract_reef_id_from_filename, create_netcdf_output,
    verify_output_structure, filter_reefs_by_bounds, get_release_times_from_netcdf,
    calculate_species_day_weights, calculate_species_hour_weights, calculate_combined_day_hour_weights
)
from spatial_metrics import calculate_angles_and_distances
from ecological_processes import piecewise_decay, piecewise_competence, connolly_competence


def main_calculations(k, particle_files, config, data_shape, n_repetitions, sample_size, cutt_off, num_sites):
    """
    Process a single reef for all bootstrap samples.
    
    Parameters
    ----------
    k : int
        Reef index (0 to num_reefs-1)
    particle_files : list
        List of particle file paths
    config : dict
        Configuration dictionary
    data_shape : gpd.GeoDataFrame
        Shapefile data
    n_repetitions : int
        Number of bootstrap repetitions
    sample_size : int
        Bootstrap sample size
    cutt_off : float
        Minimum age cutoff
    num_sites : int
        Total number of reef sites
        
    Returns
    -------
    dict
        Dictionary containing:
        - reef_id: int
        - connectivity_slice: np.ndarray (num_sites, 2, n_repetitions)
        - status: str ('success' or 'failed')
        - error: str (if failed)
        - memory_usage: float (MB)
    """
    try:
        # Get reef information first
        particle_file = particle_files[k]
        reef_id = extract_reef_id_from_filename(particle_file)
        
        # Track memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"    Reef {reef_id}: Starting with {initial_memory:.1f}MB memory")
        
        # Initialize connectivity slice for this reef
        connectivity_slice = np.zeros((num_sites, 2, n_repetitions), dtype=np.float32)
        
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
        
        # Process all bootstrap samples for this reef (limit to 2 for testing)
        max_repetitions = min(2, n_repetitions)  # Stop after 2 repetitions for testing
        for sample_idx in range(max_repetitions):
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
            
            # Filter reefs based on particle spatial bounds
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
                    connect_max_moneghetti = (connect_df.groupby('traj')['connect_moneghetti'].max() / total_trajectories).values
                    connect_max_connolly = (connect_df.groupby('traj')['connect_connolly'].max() / total_trajectories).values
                    
                    # Average connectivity across trajectories
                    connectivity_moneghetti = connect_max_moneghetti.mean()
                    connectivity_connolly = connect_max_connolly.mean()
                    
                    # Store connectivity
                    connectivity_slice[sink_reef_id, 0, sample_idx] = connectivity_moneghetti
                    connectivity_slice[sink_reef_id, 1, sample_idx] = connectivity_connolly
                    
                    # Remove settled particles to avoid double-counting
                    particles_copy = particles_copy.drop(particles_copy.index[settled_indices]).reset_index(drop=True)
            
            # Debug output after each bootstrap sample
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"    Reef {reef_id}: Sample {sample_idx + 1}/{max_repetitions} - Memory: {current_memory:.1f}MB")
        
        # Calculate memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        print(f"    Reef {reef_id}: Completed - Memory used: {memory_usage:.1f}MB, Final: {final_memory:.1f}MB")
        
        return {
            'reef_id': reef_id,
            'connectivity_slice': connectivity_slice,
            'status': 'success',
            'error': None,
            'memory_usage': memory_usage
        }
        
    except Exception as e:
        return {
            'reef_id': reef_id if 'reef_id' in locals() else k,
            'connectivity_slice': None,
            'status': 'failed',
            'error': str(e),
            'memory_usage': 0
        }


def process_reefs_in_chunks(particle_files, config, data_shape, n_repetitions, sample_size, cutt_off, num_sites, chunk_size=200):
    """
    Process reefs in chunks to manage memory and improve performance.
    
    Parameters
    ----------
    particle_files : list
        List of particle file paths
    config : dict
        Configuration dictionary
    data_shape : gpd.GeoDataFrame
        Shapefile data
    n_repetitions : int
        Number of bootstrap repetitions
    sample_size : int
        Bootstrap sample size
    cutt_off : float
        Minimum age cutoff
    num_sites : int
        Total number of reef sites
    chunk_size : int
        Number of reefs per chunk
        
    Returns
    -------
    tuple
        (all_results, memory_stats)
    """
    num_reefs = len(particle_files)
    chunks = [list(range(i, min(i + chunk_size, num_reefs))) for i in range(0, num_reefs, chunk_size)]
    
    print(f"Processing {num_reefs} reefs in {len(chunks)} chunks of {chunk_size} reefs each")
    
    # Initialize results and statistics
    all_results = []
    memory_stats = []
    failed_reefs = []
    
    # Process each chunk
    for chunk_idx, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_idx + 1}/{len(chunks)} (reefs {chunk[0]}-{chunk[-1]})")
        
        # Process reefs in this chunk in parallel
        n_jobs = int(os.getenv('SLURM_CPUS_ON_NODE', 64))
        with parallel_backend(backend='loky', n_jobs=n_jobs):
            chunk_results = Parallel()(delayed(main_calculations)(k, particle_files, config, data_shape, n_repetitions, sample_size, cutt_off, num_sites) for k in chunk)
        
        # Collect results and statistics
        chunk_memory = []
        for result in chunk_results:
            all_results.append(result)
            if result['status'] == 'success':
                chunk_memory.append(result['memory_usage'])
            else:
                failed_reefs.append(result['reef_id'])
                print(f"  ❌ Reef {result['reef_id']} failed: {result['error']}")
        
        # Calculate chunk statistics
        if chunk_memory:
            avg_memory = np.mean(chunk_memory)
            max_memory = np.max(chunk_memory)
            memory_stats.append({
                'chunk': chunk_idx + 1,
                'avg_memory_mb': avg_memory,
                'max_memory_mb': max_memory,
                'successful_reefs': len(chunk_memory),
                'failed_reefs': len(chunk) - len(chunk_memory)
            })
            print(f"  ✅ Completed chunk {chunk_idx + 1} - Avg memory: {avg_memory:.1f}MB, Max: {max_memory:.1f}MB")
        else:
            print(f"  ❌ Chunk {chunk_idx + 1} failed completely")
    
    return all_results, memory_stats, failed_reefs


def run_connectivity_analysis_parallel(config_path: str, shapefile_path: str = None, 
                                     particle_data_path: str = None, output_path: str = None,
                                     release_day: str = "2015-10-29", chunk_size: int = 200):
    """
    Run connectivity analysis using parallel processing.
    
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
    chunk_size : int
        Number of reefs per chunk.
    """
    print("="*80)
    print("REEF CONNECTIVITY ANALYSIS WITH PARALLEL PROCESSING")
    print("="*80)
    
    start_time = time.time()
    
    # 1. Load configuration
    print(f"\n1. Loading configuration from: {config_path}")
    config = load_config(config_path)
    print(f"   ✅ Configuration loaded")
    
    # Extract bootstrap parameters
    sample_size = config['bootstrap']['sample_size']
    n_repetitions = config['bootstrap']['n_repetitions']
    print(f"   Bootstrap: {sample_size} particles, {n_repetitions} repetitions")
    
    # 2. Load shapefile and calculate spatial metrics
    print(f"\n2. Loading shapefile and calculating spatial metrics...")
    if shapefile_path is None:
        shapefile_path = config['default_paths']['shapefile']
    
    data_shape, reef_centroids, num_sites = load_shapefile_and_centroids(shapefile_path)
    print(f"   ✅ Loaded {num_sites} reef sites")
    
    print(f"   Calculating spatial metrics...")
    angle_matrix, distance_matrix, direction_matrix = calculate_angles_and_distances(reef_centroids, num_sites)
    print(f"   ✅ Spatial metrics calculated: {angle_matrix.shape}")
    
    # 3. Find particle files
    print(f"\n3. Finding particle files for release day {release_day}...")
    if particle_data_path is None:
        particle_data_path = config['default_paths']['particle_data']
    print(f" \n path to particles {particle_data_path}")
    
    particle_files = list_particle_files(particle_data_path, release_day)
    print(f"Found {len(particle_files)} particle files for release day {release_day}")
    print(f"   ✅ Found {len(particle_files)} particle files")
    
    # 4. Initialize connectivity matrix
    print(f"\n4. Initializing connectivity matrix...")
    print(f"   Dimensions: {num_sites} sources × {num_sites} sinks × 2 treatments × {n_repetitions} samples")
    
    # Calculate minimum age for competence
    tc = config['competence']['tc']
    t_min = config['connolly_competence']['t_min']
    cutt_off = min(tc, t_min)
    print(f"   Minimum age cutoff: {cutt_off} days")

    connectivity_matrix = np.zeros((num_sites, num_sites, 2, n_repetitions), dtype=np.float32)
    
    # 5. Process reefs in parallel chunks
    print(f"\n5. Processing reefs in parallel chunks...")
    all_results, memory_stats, failed_reefs = process_reefs_in_chunks(
        particle_files, config, data_shape, n_repetitions, sample_size, cutt_off, num_sites, chunk_size
    )
    
    # 6. Collect results into connectivity matrix
    print(f"\n6. Collecting results into connectivity matrix...")
    successful_reefs = 0
    for result in all_results:
        if result['status'] == 'success':
            reef_id = result['reef_id']
            connectivity_slice = result['connectivity_slice']
            connectivity_matrix[reef_id, :, :, :] = connectivity_slice
            successful_reefs += 1
    
    print(f"   ✅ Collected results from {successful_reefs} successful reefs")
    if failed_reefs:
        print(f"   ❌ {len(failed_reefs)} reefs failed: {failed_reefs}")
    
    # 7. Print memory statistics
    if memory_stats:
        print(f"\n7. Memory usage statistics:")
        total_avg_memory = np.mean([stat['avg_memory_mb'] for stat in memory_stats])
        total_max_memory = np.max([stat['max_memory_mb'] for stat in memory_stats])
        print(f"   Average memory per reef: {total_avg_memory:.1f}MB")
        print(f"   Maximum memory per reef: {total_max_memory:.1f}MB")
        print(f"   Estimated memory for 64 parallel reefs: {total_max_memory * 64:.1f}MB")
    
    # 8. Create output directory
    if output_path is None:
        output_dir = config['default_paths']['output_directory']
        output_path = os.path.join(output_dir, f"connectivity_results_{release_day}.nc")
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n8. Created output directory: {output_dir}")
    
    # 9. Save results to NetCDF
    print(f"\n9. Saving results to NetCDF...")
    print(f"   Output file: {output_path}")
    
    create_netcdf_output(
        output_path, num_sites, num_sites, n_repetitions,
        angle_matrix, distance_matrix, direction_matrix, connectivity_matrix
    )
    
    # 10. Verify output
    print(f"\n10. Verifying output...")
    verification_ok = verify_output_structure(output_path, num_sites, num_sites, n_repetitions)
    
    if verification_ok:
        file_size = os.path.getsize(output_path)
        print(f"   ✅ Output verified successfully")
        print(f"   File size: {file_size:,} bytes")
    else:
        print(f"   ❌ Output verification failed")
        return False
    
    # 11. Summary
    processing_time = time.time() - start_time
    print(f"\n" + "="*80)
    print(f"ANALYSIS COMPLETE")
    print(f"="*80)
    print(f"✅ Processing time: {processing_time:.2f} seconds")
    print(f"✅ Input files: {len(particle_files)} particle files")
    print(f"✅ Processed files: {successful_reefs}")
    print(f"✅ Failed files: {len(failed_reefs)}")
    print(f"✅ Output: {output_path}")
    print(f"✅ Matrix dimensions: {num_sites}×{num_sites}×2 treatments×{n_repetitions} samples")
    
    return True


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Run reef connectivity analysis using parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_connectivity_parallel.py
  python run_connectivity_parallel.py --config config/connectivity_parameters.yaml
  python run_connectivity_parallel.py --chunk-size 100 --n-jobs 32
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
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=200,
        help='Number of reefs per chunk (default: 200)'
    )
    
    args = parser.parse_args()
    
    # Run the analysis
    success = run_connectivity_analysis_parallel(
        config_path=args.config,
        shapefile_path=args.shapefile,
        particle_data_path=args.particle_data,
        output_path=args.output,
        release_day=args.release_day,
        chunk_size=args.chunk_size
    )
    
    if success:
        print("\n🎉 Analysis completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
