"""
Input/Output utilities for reef connectivity calculations.

This module provides functions for reading and writing data files:
- Shapefile operations (reading reef polygons)
- NetCDF file operations (particle data and results)
- Configuration file handling
"""

import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import os
from pathlib import Path
import yaml


def load_shapefile_and_centroids(shapefile_path: str) -> Tuple[gpd.GeoDataFrame, List, int]:
    """
    Load shapefile and calculate reef centroids.
    
    Parameters
    ----------
    shapefile_path : str
        Path to the shapefile.
        
    Returns
    -------
    tuple
        (data_shape, reef_centroids, num_sites)
    """
    print(f"Loading shapefile: {shapefile_path}")
    data_shape = gpd.read_file(shapefile_path)
    num_sites = data_shape.shape[0]
    
    # Add min/max lat and lon columns for optimization
    data_shape['min_lat'] = data_shape.geometry.bounds['miny']
    data_shape['max_lat'] = data_shape.geometry.bounds['maxy']
    data_shape['min_lon'] = data_shape.geometry.bounds['minx']
    data_shape['max_lon'] = data_shape.geometry.bounds['maxx']
    
    reef_centroids = []
    ## getting the centroid's location
    for site in range(0, num_sites):
        ## release reef
        value_index = list(data_shape.loc[data_shape['FID'] == site].index)
        value_index = int("".join(map(str, value_index)))
        polygon = data_shape['geometry'][value_index]
        reef_centroids.append(polygon.centroid)
    
    print(f"Loaded {num_sites} reef sites")
    return data_shape, reef_centroids, num_sites


def load_particle_data(file_path: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Load particle data from NetCDF file and calculate spatial bounds.
    
    Parameters
    ----------
    file_path : str
        Path to the particle NetCDF file.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        DataFrame with particle data and spatial bounds dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Particle file not found: {file_path}")
    
    print(f"Loading particle data: {file_path}")
    output_nc = xr.open_dataset(file_path)
    
    particles = pd.DataFrame({
        'latitudes': output_nc['lat'].values.ravel(),
        'longitudes': output_nc['lon'].values.ravel(),
        'trajectories': output_nc['trajectory'].values.ravel(),
        'age': output_nc['age'].values.ravel() / 86400  # Seconds to days
    })
    
    ntraj = output_nc.dims['traj']
    output_nc.close()
    
    # Clean the nans
    particles = particles.dropna()
    
    # Calculate spatial bounds
    spatial_bounds = {
        'min_lat': particles['latitudes'].min(),
        'max_lat': particles['latitudes'].max(),
        'min_lon': particles['longitudes'].min(),
        'max_lon': particles['longitudes'].max()
    }
    
    return particles, spatial_bounds


def load_sampled_trajectory_data(file_path: str, trajectory_indices: np.ndarray, cutt_off: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Load only sampled trajectories from NetCDF file to reduce memory usage.
    
    Parameters
    ----------
    file_path : str
        Path to the particle NetCDF file.
    trajectory_indices : np.ndarray
        Indices of trajectories to sample.
    cutt_off : float
        Minimum age for competence.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        DataFrame with all particles from sampled trajectories and spatial bounds dictionary.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Particle file not found: {file_path}")
    
    print(f"Loading sampled trajectory data: {file_path} (trajectories: {len(trajectory_indices)})")
    output_nc = xr.open_dataset(file_path)
    
    # Sample trajectories and get all particles from those trajectories
    # NetCDF structure: [time, traj] dimensions
    sampled_lat = output_nc['lat'].values[trajectory_indices, :].ravel()
    sampled_lon = output_nc['lon'].values[trajectory_indices, :].ravel()
    sampled_age = output_nc['age'].values[trajectory_indices, :].ravel()
    trajectory_ids = output_nc['trajectory'].values[trajectory_indices, :].ravel()
    
    particles = pd.DataFrame({
        'latitudes': sampled_lat,
        'longitudes': sampled_lon,
        'trajectories': trajectory_ids,
        'age': sampled_age / 86400  # Seconds to days
    })
    ## filter particles below minimum age
    particles = particles[particles['age'] > cutt_off]
    output_nc.close()
    
    # Clean the nans
    particles = particles.dropna()
    
    # Calculate spatial bounds from sampled trajectories
    spatial_bounds = {
        'min_lat': particles['latitudes'].min(),
        'max_lat': particles['latitudes'].max(),
        'min_lon': particles['longitudes'].min(),
        'max_lon': particles['longitudes'].max()
    }
    
    print(f"     Loaded {len(particles)} particles from {len(trajectory_indices)} trajectories")
    
    return particles, spatial_bounds


def filter_reefs_by_bounds(data_shape: gpd.GeoDataFrame, spatial_bounds: Dict[str, float]) -> List[int]:
    """
    Filter reefs based on spatial bounds to reduce computational load.
    
    Parameters
    ----------
    data_shape : gpd.GeoDataFrame
        GeoDataFrame containing reef polygons.
    spatial_bounds : Dict[str, float]
        Dictionary with 'min_lat', 'max_lat', 'min_lon', 'max_lon' keys.
        
    Returns
    -------
    List[int]
        List of reef indices that fall within the spatial bounds.
    """
    # Create boolean masks for latitude and longitude bounds
    lat_mask = (data_shape['min_lat'] <= spatial_bounds['max_lat']) & \
               (data_shape['max_lat'] >= spatial_bounds['min_lat'])
    
    lon_mask = (data_shape['min_lon'] <= spatial_bounds['max_lon']) & \
               (data_shape['max_lon'] >= spatial_bounds['min_lon'])
    
    # Combine masks
    bounds_mask = lat_mask & lon_mask
    
    # Get FID values of reefs within bounds (instead of indices)
    reef_fids = data_shape['FID'].iloc[np.where(bounds_mask)[0]].tolist()
    return reef_fids


def list_particle_files(data_directory: str, release_day: str) -> List[str]:
    """
    List all particle files for a given release day.
    
    Parameters
    ----------
    data_directory : str
        Directory containing particle files.
    release_day : str
        Release day string (e.g., "365").
        
    Returns
    -------
    List[str]
        List of particle file paths.
    """
    pattern = f"/{release_day}/GBR1_H2p0_Coral_Release_{release_day}_Polygon_*_Wind_3_percent_displacement_field.nc"
    data_path = Path(data_directory)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_directory}")
    
    files = list(data_path.glob(pattern))
    files.sort()  # Sort to ensure consistent ordering
    
    print(f"Found {len(files)} particle files for release day {release_day}")
    return [str(f) for f in files]


def extract_reef_id_from_filename(filename: str) -> int:
    """
    Extract reef ID from particle filename.
    
    Parameters
    ----------
    filename : str
        Particle filename.
        
    Returns
    -------
    int
        Reef ID extracted from filename.
    """
    # Extract reef ID from filename like: ...Polygon_1234_Wind...
    base_name = os.path.basename(filename)
    parts = base_name.split('_')
    
    for i, part in enumerate(parts):
        if part == 'Polygon' and i + 1 < len(parts):
            return int(parts[i + 1])
    
    raise ValueError(f"Could not extract reef ID from filename: {filename}")


def create_netcdf_output(output_path: str, num_sources: int, num_sinks: int, num_samples: int,
                        angle_data: np.ndarray, distance_data: np.ndarray, 
                        direction_data: np.ndarray, connectivity_data: np.ndarray) -> None:
    """
    Create NetCDF file with connectivity results using float32 and compression.
    
    Parameters
    ----------
    output_path : str
        Path for output NetCDF file.
    num_sources : int
        Number of source reefs.
    num_sinks : int
        Number of sink reefs.
    num_samples : int
        Number of bootstrap samples.
    angle_data : np.ndarray
        Angle matrix [source, sink].
    distance_data : np.ndarray
        Distance matrix [source, sink].
    direction_data : np.ndarray
        Direction matrix [source, sink].
    connectivity_data : np.ndarray
        Connectivity matrix [source, sink, treatment, sample].
    """
    print(f"Creating NetCDF output: {output_path}")
    
    # Convert data to float32 for size reduction
    angle_data_f32 = angle_data.astype(np.float32)
    distance_data_f32 = distance_data.astype(np.float32)
    direction_data_f32 = direction_data.astype(np.float32)
    connectivity_data_f32 = connectivity_data.astype(np.float32)
    
    # Create xarray Dataset with float32 data
    ds = xr.Dataset(
        {
            'angle': (['source', 'sink'], angle_data_f32),
            'distance': (['source', 'sink'], distance_data_f32),
            'direction': (['source', 'sink'], direction_data_f32),
            'connectivity': (['source', 'sink', 'treatment', 'sample'], connectivity_data_f32)
        },
        coords={
            'source': np.arange(num_sources),
            'sink': np.arange(num_sinks),
            'treatment': ['moneghetti', 'connolly'],
            'sample': np.arange(num_samples)
        }
    )
    
    # Add attributes
    ds.attrs['title'] = 'Reef Connectivity Analysis Results'
    ds.attrs['description'] = 'Bootstrap-sampled connectivity matrices with spatial metrics'
    ds.attrs['created_by'] = 'ReefConnect Process 1'
    ds.attrs['data_type'] = 'float32'
    ds.attrs['compression'] = 'zlib level 7'
    
    # Variable attributes
    ds['angle'].attrs = {
        'long_name': 'Angle between source and sink reefs',
        'units': 'degrees',
        'description': 'Angle from source to sink reef centroid',
        'dtype': 'float32'
    }
    
    ds['distance'].attrs = {
        'long_name': 'Distance between source and sink reefs',
        'units': 'kilometers',
        'description': 'Great circle distance between reef centroids',
        'dtype': 'float32'
    }
    
    ds['direction'].attrs = {
        'long_name': 'Direction sector from source to sink',
        'units': 'sector_number',
        'description': 'Cardinal direction sector (0-35, 10-degree sectors)',
        'dtype': 'float32'
    }
    
    ds['connectivity'].attrs = {
        'long_name': 'Connectivity between source and sink reefs',
        'units': 'probability',
        'description': 'Bootstrap-sampled connectivity values for different competence treatments',
        'dtype': 'float32',
        'treatments': 'moneghetti: Piecewise Weibull-Exponential competence model, connolly: Connolly et al. 2010 competence model'
    }
    
    # Save to NetCDF with compression
    encoding = {
        'angle': {'dtype': 'float32', 'zlib': True, 'complevel': 7},
        'distance': {'dtype': 'float32', 'zlib': True, 'complevel': 7},
        'direction': {'dtype': 'float32', 'zlib': True, 'complevel': 7},
        'connectivity': {'dtype': 'float32', 'zlib': True, 'complevel': 7}
    }
    
    ds.to_netcdf(output_path, engine='netcdf4', encoding=encoding)
    print(f"NetCDF file saved: {output_path}")
    print(f"Data type: float32, Compression: zlib level 7")


def load_config(config_path: str = "config/connectivity_parameters.yaml") -> Dict[str, Any]:
    """
    Load configuration parameters from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.
        
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print(f"Configuration loaded from: {config_path}")
    return config


def get_release_times_from_netcdf(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract release times for all trajectories from NetCDF file.
    
    Parameters
    ----------
    file_path : str
        Path to the particle NetCDF file.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (release_times, trajectory_indices) where release_times are datetime64 objects
        and trajectory_indices are the corresponding trajectory indices.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Particle file not found: {file_path}")
    
    output_nc = xr.open_dataset(file_path)
    
    # Get release times (first observation for each trajectory)
    release_times = output_nc.time.isel(obs=0).values  # First time point for each trajectory
    
    # Convert to UTC+10 timezone
    release_times_utc10 = release_times + np.timedelta64(10, 'h')
    
    trajectory_indices = np.arange(output_nc.sizes['traj'])
    
    output_nc.close()
    
    return release_times_utc10, trajectory_indices


def calculate_species_day_weights(release_times: np.ndarray, species: str = 'acropora') -> np.ndarray:
    """
    Calculate day-based probability weights for trajectory sampling based on species.
    
    Parameters
    ----------
    release_times : np.ndarray
        Array of release times (datetime64 objects).
    species : str
        Species name ('acropora' or 'merulinidae').
        
    Returns
    -------
    np.ndarray
        Probability weights for each trajectory (sums to 1).
    """
    # Convert to days since first release
    first_release = np.min(release_times)
    days_since_first = (release_times - first_release) / np.timedelta64(1, 'D')
    release_days = np.floor(days_since_first).astype(int)
    
    # Create weights array
    weights = np.ones(len(release_times))
    
    if species.lower() == 'acropora':
        # Specific probabilities for each day
        day_probabilities = [0.031055901, 0.2111180124, 0.3975155528, 0.186335404, 0.173913043]
        for i, day in enumerate(release_days):
            weights[i] = day_probabilities[day]
                
    elif species.lower() == 'merulinidae':
        # Specific probabilities for each day
        day_probabilities = [0, 0.012195122, 0.0243902439, 0.3292682927, 0.6341463415]
        for i, day in enumerate(release_days):
            weights[i] = day_probabilities[day]
    else:
        raise ValueError(f"Unknown species: {species}. Must be 'acropora' or 'merulinidae'")
    
    # Normalize so weights sum to 1
    weights = weights / np.sum(weights)
    
    return weights


def calculate_species_hour_weights(release_times: np.ndarray, species: str = 'acropora') -> np.ndarray:
    """
    Calculate hour-based probability weights for trajectory sampling based on species.
    
    This function implements family-specific spawning time distributions where:
    - Hour 20 = spawning between 20:00-21:00 (8-9 PM)
    - Hour 24 = spawning between 00:00-01:00 (midnight-1 AM) on following day
    - Timezone assumed to be UTC+10
    
    Parameters
    ----------
    release_times : np.ndarray
        Array of release times (datetime64 objects).
    species : str
        Species name ('acropora' or 'merulinidae').
        
    Returns
    -------
    np.ndarray
        Probability weights for each trajectory (sums to 1).
    """
    # Time distribution probabilities by family
    time_probs = {
        "default": {
            20: 0.2, 21: 0.2, 22: 0.2, 23: 0.2, 24: 0.2
        },
        "acropora": {
            20: 0.253598355037697,
            21: 0.328992460589445,
            22: 0.368974183230523,
            23: 0.0242175005711675,
            24: 0.0242175005711675,
        },
        "merulinidae": {
            20: 0.334925476204178,
            21: 0.234793117339011,
            22: 0.385567128963573,
            23: 0.0391321862231686,
            24: 0.00558209127006963,
        },
    }
    
    # Extract hour from UTC+10 times
    # Extract hour (0-23, where 24 becomes 0 for next day)
    hours = release_times.astype('datetime64[h]').astype(int) % 24
    
    # Convert hour 0 to hour 24 (midnight-1 AM on following day)
    hours = np.where(hours == 0, 24, hours)
    
    # Get probability distribution for this species
    species_key = species.lower()
    if species_key not in time_probs:
        species_key = "default"
    
    hour_probs = time_probs[species_key]
    
    # Create weights array
    weights = np.ones(len(release_times))
    
    # Apply hour-based probabilities
    for hour, prob in hour_probs.items():
        mask = hours == hour
        weights[mask] = prob
    
    # Normalize so weights sum to 1
    weights = weights / np.sum(weights)
    
    return weights


def calculate_combined_day_hour_weights(release_times: np.ndarray, species: str = 'acropora') -> np.ndarray:
    """
    Calculate combined day and hour probability weights for trajectory sampling.
    
    This function combines both day-based and hour-based probability weights by multiplying them.
    
    Parameters
    ----------
    release_times : np.ndarray
        Array of release times (datetime64 objects).
    species : str
        Species name ('acropora' or 'merulinidae').
        
    Returns
    -------
    np.ndarray
        Combined probability weights for each trajectory (sums to 1).
    """
    # Get day-based weights
    day_weights = calculate_species_day_weights(release_times, species)
    
    # Get hour-based weights
    hour_weights = calculate_species_hour_weights(release_times, species)
    
    # Combine by multiplying (assuming independence)
    combined_weights = day_weights * hour_weights
    
    # Normalize so weights sum to 1
    combined_weights = combined_weights / np.sum(combined_weights)
    
    return combined_weights


def verify_output_structure(output_path: str, expected_sources: int, 
                          expected_sinks: int, expected_samples: int) -> bool:
    """
    Verify the structure of the output NetCDF file.
    
    Parameters
    ----------
    output_path : str
        Path to NetCDF file to verify.
    expected_sources : int
        Expected number of source reefs.
    expected_sinks : int
        Expected number of sink reefs.
    expected_samples : int
        Expected number of samples.
        
    Returns
    -------
    bool
        True if structure is correct.
    """
    try:
        ds = xr.open_dataset(output_path)
        
        # Check dimensions
        assert ds.dims['source'] == expected_sources, f"Wrong number of sources: {ds.dims['source']}"
        assert ds.dims['sink'] == expected_sinks, f"Wrong number of sinks: {ds.dims['sink']}"
        assert ds.dims['treatment'] == 2, f"Wrong number of treatments: {ds.dims['treatment']}"
        assert ds.dims['sample'] == expected_samples, f"Wrong number of samples: {ds.dims['sample']}"
        
        # Check treatment labels
        expected_treatments = ['moneghetti', 'connolly']
        actual_treatments = ds.coords['treatment'].values.tolist()
        assert actual_treatments == expected_treatments, f"Wrong treatment labels: {actual_treatments}"
        
        # Check variables exist
        required_vars = ['angle', 'distance', 'direction', 'connectivity']
        for var in required_vars:
            assert var in ds.variables, f"Missing variable: {var}"
        
        ds.close()
        print(f"✅ Output file structure verified: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Output file verification failed: {e}")
        return False 
