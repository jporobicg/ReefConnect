import sys
import os
from reefconnect.connectivity import ConnectivityCalculator
import geopandas as gpd
import numpy as np
import time
import xarray as xr
from joblib import Parallel, delayed, parallel_backend

def main(release_start_day):
    ## ~~~~~~~~~~~~~~~~~~~~ ##
    ## ~      Parameters  ~ ##
    ## ~~~~~~~~~~~~~~~~~~~~ ##
    ## Parameters used in the decay and competence function
    ## Decay
    Tcp_decay = 2.583
    lmbda1    = 0.4
    lmbda2    = 0.019
    v1        = 2.892
    v2        = 1.716
    sigma1    = 0
    sigma2    = 0

    ## Competence
    tc       = 3.333
    Tcp_comp = 69.91245
    alpha    = 1.295
    beta1    = 0.001878001
    beta2    = 0.3968972
    v        = 0.364

    ## ~~~~~~~~~~~~~~~ ##
    ## ~   Main Code ~ ##
    ## ~~~~~~~~~~~~~~~ ##
    shapefile = '/datasets/work/oa-coconet/work/oceanparcels_gbr_Coral/Shape_files/gbr1_coral_1m_merged_buffer0p001.shp'
    data_shape = gpd.read_file(shapefile)

    ## getting the boundaries of each reefs for
    ## entired GBR
    num_reefs = data_shape.shape[0]
    min_lat = []; max_lat = []
    for i_polygon in range(0, num_reefs):
        min_lat.append(np.nanmin(np.array(data_shape['geometry'][i_polygon].bounds)[[1,3]]))
        max_lat.append(np.nanmax(np.array(data_shape['geometry'][i_polygon].bounds)[[1,3]]))
    data_shape['min_lat']= min_lat
    data_shape['max_lat']= max_lat

    path='/datasets/work/oa-coconet/work/OceanParcels_outputs/Coral/' + release_start_day
    jobs = range(num_reefs)
    n_jobs = int(os.getenv('SLURM_CPUS_ON_NODE', 10))

    with parallel_backend(backend='loky', n_jobs=n_jobs):
        results_list = Parallel()(delayed(ConnectivityCalculator)(k) for k in jobs)

    print('calculations done', time.strftime("%H:%M:%S"), flush = True)

    ## Creating empty arrays
    connectivity_matrix_sum   = np.empty((num_reefs, num_reefs))
    connectivity_matrix_max   = np.empty((num_reefs, num_reefs))
    connectivity_variance_sum = np.zeros((num_reefs, num_reefs))
    connectivity_variance_max = np.zeros((num_reefs, num_reefs))

    for k in jobs:
        connectivity_matrix_max[k,:] = results_list[k][0]
        connectivity_matrix_sum[k,:] = results_list[k][1]
        connectivity_variance_max[k,:] = results_list[k][2]
        connectivity_variance_sum[k,:] = results_list[k][3]

    # Create coordinates for the matrices
    reef_coords = range(num_reefs)

    # Create DataArrays with proper dimensions
    ds = xr.Dataset(
        data_vars={
            'connectivity_sum': xr.DataArray(
                connectivity_matrix_sum,
                dims=['source_reef', 'destination_reef'],
                coords={'source_reef': reef_coords, 'destination_reef': reef_coords}
            ),
            'connectivity_max': xr.DataArray(
                connectivity_matrix_max,
                dims=['source_reef', 'destination_reef'],
                coords={'source_reef': reef_coords, 'destination_reef': reef_coords}
            ),
            'variance_sum': xr.DataArray(
                connectivity_variance_sum,
                dims=['source_reef', 'destination_reef'],
                coords={'source_reef': reef_coords, 'destination_reef': reef_coords}
            ),
            'variance_max': xr.DataArray(
                connectivity_variance_max,
                dims=['source_reef', 'destination_reef'],
                coords={'source_reef': reef_coords, 'destination_reef': reef_coords}
            )
        }
    )

    # Add metadata
    ds.attrs['creation_date'] = time.strftime("%Y-%m-%d")
    ds.attrs['description'] = 'Connectivity matrices for coral reef larvae dispersal'

    # Save to NetCDF file
    output_file = f"{path}/{release_start_day}_connectivity_matrices.nc"
    ds.to_netcdf(output_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_connectivity.py <release_start_day>")
        sys.exit(1)
    main(sys.argv[1])
