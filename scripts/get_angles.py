import geopandas as gpd
import numpy as np
from ..angles import calculate_angle, calculate_direction_sector
from ..utils import haversine
import xarray as xr

def process_reef_angles(shapefile_path, output_dir='output'):
    """Process reef angles and generate output files.
    
    Args:
        shapefile_path (str): Path to the input shapefile
        output_dir (str): Directory for output files
    """
    # Read shapefile
    data_shape = gpd.read_file(shapefile_path)
    num_sites = data_shape.shape[0]
    
    # Get centroids
    reef_centroids = [
        data_shape['geometry'][int("".join(map(str, data_shape.loc[data_shape['FID'] == site].index)))].centroid
        for site in range(num_sites)
    ]
    
    # Initialize arrays for results
    angles = np.zeros((num_sites, num_sites))
    directions = np.zeros((num_sites, num_sites))
    distances = np.zeros((num_sites, num_sites))
    
    # Reference vector (North)
    north_vector = [0, 1]
    
    # Process each release site
    for release_site in range(num_sites):
     
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
    
    # Save to NetCDF file
    ds.to_netcdf(f'{output_dir}/GBR_reefs_connectivity.nc')

def main():
    """Command line interface for processing reef angles."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process reef angles from shapefile.')
    parser.add_argument('--shapefile', required=True, help='Path to input shapefile')
    parser.add_argument('--output-dir', default='output', help='Output directory')
    
    args = parser.parse_args()
    process_reef_angles(args.shapefile, args.output_dir)

if __name__ == "__main__":
    main()
    
shapefile_path = '/datasets/work/oa-coconet/work/oceanparcels_gbr_Coral/Shape_files/gbr1_coral_1m_merged_buffer0p001.shp'
output_dir = '/datasets/work/oa-coconet/work/Outputs_new_Runs/Angles/'
process_reef_angles(shapefile_path, output_dir)
