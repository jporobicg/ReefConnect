"""
Main connectivity calculation function.

This module contains the main calc() function that processes particle tracking data
and calculates connectivity between reefs.
"""

import numpy as np
import pandas as pd
import xarray as xr
import time
import os

from ecological_processes import piecewise_decay, piecewise_competence, points_in_polygon


def calc(source_reef, data_shape, particle_data_path, release_start_day, 
         Tcp_decay, lmbda1, lmbda2, v1, v2, sigma1, sigma2,
         tc, Tcp_comp, alpha, beta1, beta2, v):
    """
    Calculates the connectivity metrics for a given source reef using particle dispersal tracks.

    Args:
        source_reef (int): The index of the source reef to calculate connectivity from.
        data_shape: GeoDataFrame with reef data.
        particle_data_path (str): Path to particle data directory.
        release_start_day (str): Release day string.
        Tcp_decay, lmbda1, lmbda2, v1, v2, sigma1, sigma2: Decay parameters.
        tc, Tcp_comp, alpha, beta1, beta2, v: Competence parameters.

    Returns:
        list: A list containing four numpy arrays representing the connectivity metrics:
              - connectivity_matrix_max: Maximum connectivity values for each reef in the study area.
              - connectivity_matrix_sum: Sum of connectivity values for each reef in the study area.
              - connectivity_variance_max: Variance of maximum connectivity values for each reef.
              - connectivity_variance_sum: Variance of sum of connectivity values for each reef.
    """
    print('starting Source reef: ' + str(source_reef))
    t = time.time()
    
    # Get number of reefs
    num_reefs = data_shape.shape[0]
    
    # Creating empty arrays
    connectivity_matrix_sum = np.zeros(num_reefs)
    connectivity_matrix_max = np.zeros(num_reefs)
    connectivity_variance_sum = np.zeros(num_reefs)
    connectivity_variance_max = np.zeros(num_reefs)

    file_name = f"{particle_data_path}/GBR1_H2p0_Coral_Release_{release_start_day}_Polygon_{source_reef}_Wind_3_percent_displacement_field.nc"
    
    if not os.path.exists(file_name):
        print('file missing - ' + str(source_reef))
    else:
        print(f"Trying to open: {file_name}", flush=True)
        try:
            output_nc = xr.open_dataset(file_name)
        except ValueError as e:
            print(f"Failed to open {file_name} -- error: {e}", flush=True)
            raise    
        
        ntraj = output_nc.dims['traj']
        particles = pd.DataFrame({
            'latitudes': output_nc['lat'].values.ravel(),
            'longitudes': output_nc['lon'].values.ravel(),
            'trajectories': output_nc['trajectory'].values.ravel(),
            'age': output_nc['age'].values.ravel() / 86400  # Seconds to days
        })
        output_nc.close()
        
        # Cleaning the nans
        particles = particles.dropna()
        # remove particles below minimum age
        particles = particles[particles['age'] > tc]
        
        # set particles boundaries in model domain
        # this avoids the overload of looking over all the reefs
        particle_max_lat = np.nanmax(particles['latitudes'].values)
        particle_min_lat = np.nanmin(particles['latitudes'].values)
        
        # making boolean series
        upper_bound = data_shape['min_lat'] <= particle_max_lat
        mmax = upper_bound.to_numpy()
        inf_bound = data_shape['max_lat'] >= particle_min_lat
        minf = inf_bound.to_numpy()
        boundary_reefs = np.where(np.multiply(minf, mmax))[0]

        for sink_reef in boundary_reefs:
            reef_index = data_shape['FID'][sink_reef]
            polygon = np.array(list(data_shape['geometry'][sink_reef].exterior.coords))
            
            if particles.size == 0:
                break  # breaking the loop if not more particles
                
            # Are these points inside the polygon?
            p = points_in_polygon(
                particles['longitudes'].values,
                particles['latitudes'].values,
                data_shape.min_lat[sink_reef],
                data_shape.max_lat[sink_reef],
                polygon,
            )
            m = np.where(p)[0]
            
            if len(m > 0):
                settled_age = particles['age'].iloc[m].values
                settled_traj = particles['trajectories'].iloc[m].values
                decay = piecewise_decay(settled_age, Tcp_decay, lmbda1, lmbda2, v1, v2, sigma1, sigma2)
                competence = piecewise_competence(settled_age, tc, Tcp_comp, alpha, beta1, beta2, v)
                connect = pd.DataFrame({'connect': np.array(decay) * np.array(competence), 'traj': settled_traj})
                connect_sum = (connect.groupby('traj')['connect'].sum() / ntraj).to_numpy()
                connect_max = (connect.groupby('traj')['connect'].max() / ntraj).to_numpy()
                
                # Average connectivity through all the particles in that reef
                connectivity_matrix_max[reef_index] = connect_max.mean()
                connectivity_matrix_sum[reef_index] = connect_sum.mean()
                # Variance
                connectivity_variance_max[reef_index] = connect_max.var()
                connectivity_variance_sum[reef_index] = connect_sum.var()
                particles.drop(index=particles.iloc[m].index, inplace=True)

    print(source_reef, int(time.time() - t + 0.5), flush=True)

    return [connectivity_matrix_max, connectivity_matrix_sum, connectivity_variance_max, connectivity_variance_sum] 