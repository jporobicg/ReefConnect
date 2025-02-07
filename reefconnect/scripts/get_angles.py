import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from reefconnect.angles import calculate_angle, calculate_direction_sector
from reefconnect.utils import haversine
import geopandas as gpd
import numpy as np
import xarray as xr

def process_reef_angles(shapefile_path, output_dir='output', verbose=False):
    """Process reef angles and generate output files.
    
    Args:
        shapefile_path (str): Path to the input shapefile
        output_dir (str): Directory for output files
        verbose (bool): If True, print debug information
    """
    if verbose:
        print(f"Reading shapefile from: {shapefile_path}")
    
    # Read shapefile
    data_shape = gpd.read_file(shapefile_path)
    num_sites = data_shape.shape[0]
    
    if verbose:
        print(f"Found {num_sites} reef sites")
        print("Calculating centroids...")
    
    # Get centroids
    reef_centroids = [
        data_shape['geometry'][int("".join(map(str, data_shape.loc[data_shape['FID'] == site].index)))].centroid
        for site in range(num_sites)
    ]
    
    if verbose:
        print("Initializing calculation arrays...")
    
    # Initialize arrays for results
    angles = np.zeros((num_sites, num_sites))
    directions = np.zeros((num_sites, num_sites))
    distances = np.zeros((num_sites, num_sites))
    
    # Reference vector (North)
    north_vector = [0, 1]
    
    # Process each release site
    for release_site in range(num_sites):
        if verbose and release_site % 100 == 0:  # Print progress every 100 sites
            print(f"Processing release site {release_site}/{num_sites}")
            
        for target_site in range(num_sites):
            if release_site != target_site:
                source_coords = np.array([reef_centroids[release_site].coords[0][0], 
                                        reef_centroids[release_site].coords[0][1]])
                target_coords = np.array([reef_centroids[target_site].coords[0][0], 
                                        reef_centroids[target_site].coords[0][1]])
                
                vector = [target_coords[0] - source_coords[0], 
                         target_coords[1] - source_coords[1]]
                
                angle = calculate_angle(north_vector, vector)
                if vector[0] < 0:
                    angle = 360 - angle
                    
                direction = calculate_direction_sector(angle)
                distance = haversine(source_coords, target_coords)
                
                angles[release_site, target_site] = angle
                directions[release_site, target_site] = direction
                distances[release_site, target_site] = distance
    
    if verbose:
        print("Creating xarray Dataset...")
    
    # Create xarray Dataset
    reef_ids = [int("".join(map(str, data_shape.loc[data_shape['FID'] == site].index))) for site in range(num_sites)]
    ds = xr.Dataset(
        {
            'angle': (['source', 'target'], angles),
            'direction': (['source', 'target'], directions),
            'distance': (['source', 'target'], distances),
        },
        coords={
            'source': reef_ids,
            'target': reef_ids,
        }
    )
    
    # Add variable attributes
    ds.angle.attrs['units'] = 'degrees'
    ds.angle.attrs['long_name'] = 'Angle between reefs relative to North'
    ds.direction.attrs['units'] = 'sector'
    ds.direction.attrs['long_name'] = 'Direction sector between reefs'
    ds.distance.attrs['units'] = 'kilometers'
    ds.distance.attrs['long_name'] = 'Distance between reef centroids'
    
    # Add global attributes
    from datetime import datetime
    ds.attrs['title'] = 'Great Barrier Reef Connectivity Metrics'
    ds.attrs['description'] = (
        'This dataset contains calculations of angles, directions, and distances between reef centroids '
        'in the Great Barrier Reef Marine Park. For each pair of reefs, the angle (relative to North), '
        'direction sector, and distance are calculated using the centroid positions of the reefs.'
    )
    ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d')
    ds.attrs['reef_definition_source'] = 'Great Barrier Reef Marine Park Authority (GBRMPA) reef shapefile'
    ds.attrs['methodology'] = (
        'Calculations are performed between reef centroids. Angles are calculated relative to North (0°), '
        'with angles increasing clockwise. Distances are calculated using the haversine formula. '
        'Direction sectors are derived from the calculated angles.'
    )
    ds.attrs['contact'] = 'Your Institution/Contact Information'
    
    if verbose:
        print(f"Saving NetCDF file to: {output_dir}/GBR_reefs_connectivity.nc")
    # Save to NetCDF file
    ds.to_netcdf(f'{output_dir}/GBR_reefs_connectivity.nc')

def main():
    """Command line interface for processing reef angles."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process reef angles from shapefile.')
    parser.add_argument('--shapefile', required=True, help='Path to input shapefile')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    process_reef_angles(args.shapefile, args.output_dir, args.verbose)

if __name__ == "__main__":
    main()
    


