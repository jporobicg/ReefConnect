import numpy as np
import glob
from .utils import get_centroids, reef_ordering, select_sink_locations
from .kernels import (remove_outliers, count_sectors, connectivity_by_sectors,
                     select_top_sectors, find_two_groups, find_positions, calculate_ds)
# Load input files for connectivity, angle, and shape information
# Step 1: Load shapefile and get centroids and reef order
shapefile = '/datasets/work/oa-coconet/work/oceanparcels_gbr_Coral/Shape_files/gbr1_coral_1m_merged_buffer0p001.shp'
centroids, reef_order = get_centroids(shapefile)

# Step 2: Load distance and angle matrices
distance_matrix = np.loadtxt('/datasets/work/oa-coconet/work/gbr_connectivity/output/GBR_reefs_distance.csv',
                             delimiter=',', skiprows=1, usecols=range(1, len(reef_order)+1))
angle_matrix = np.loadtxt('/datasets/work/oa-coconet/work/gbr_connectivity/output/GBR_reefs_angles.csv',
                          delimiter=',', skiprows=1, usecols=range(1, len(reef_order)+1))


# Step 3: Main loop over the connectivity files

files_connect = glob.glob(
    '/datasets/work/oa-coconet/work/OceanParcels_outputs/Coral/Connectivity_matrix/*_Connectivity_max.csv.y')
for f in range(0, len(files_connect)):
    period = files_connect[f][-33:-23]
    connectivity_matrix = np.loadtxt(files_connect[f], delimiter=',')
    distance_matrix_ordered, angle_matrix_ordered = reef_ordering(
        distance_matrix, angle_matrix, reef_order)
    # Main code:
    # Step 4: Open output file for writing kernel parameters
    Kernel_outFile = open(str(period) + 'Kernel_parameters_Corals.csv', 'w')
    Kernel_outFile.write("reef_ID, lat, lon, connectivity_sector_01, S_sector_01, DS_sector_01, Distance_sector_01, Proportion_reefs_01,connectivity_sector_02, S_sector_02, DS_sector_02, Distance_sector_02, Proportion_reefs_02\n")
    # Step 5: Iterate over sink areas
    for source_area in range(0, len(connectivity_matrix[0])):
        Kernel_outFile.write(str(
            source_area) + ',' + str(centroids[source_area].y) + ',' + str(centroids[source_area].x) + ',')
        # Step 6: Select source locations based on connectivity, angle, and distance matrices
        source_connect, source_angle, source_distance = select_sink_locations(
            connectivity_matrix, angle_matrix_ordered, distance_matrix_ordered, source_area)
        # Step 7: Remove outliers from the selected sources
        filter_connect, filter_angle, filter_centroid, filter_distance = remove_outliers(
            source_connect, source_angle, centroids, source_distance, percent=5)
        # Step 8: Count sectors based on filtered angle values
        sectors_array = count_sectors(filter_angle)
        # Step 9: Calculate total connectivity, bandwidth, average distance, and weighted distance by sectors
        total_connectivity, bandwidth, avg_distance, wgt_distance, effective_reefs, total_reefs = connectivity_by_sectors(
            sectors_array, filter_connect, filter_distance)
        # Step 10: Check if total connectivity is zero, write 'nan' values, and continue to the next sink area
        if (sum(total_connectivity) == 0):
            Kernel_outFile.write('nan,nan,nan,nan,nan,nan,nan,nan,nan,nan\n')
            continue
        n_sectors = 2  # number of sectors
        # Step 11: Select top sectors based on total connectivity
        selected_sector = select_top_sectors(total_connectivity, n_sectors)
        # Step 12: Calculate proportion of recruitment and add additional sectors if needed
        prop_recruitment = sum(
            total_connectivity[selected_sector])/sum(total_connectivity)
        while prop_recruitment < 0.8:
            selected = select_boundary_sectors(
                selected_sector, total_connectivity)
            selected_sector = np.append(selected_sector, selected)
            prop_recruitment = sum(
                total_connectivity[selected_sector])/sum(total_connectivity)
        # 12.5 cleaning the vector with zeros
        selected_sector = selected_sector[np.where(
            total_connectivity[selected_sector])[0]]
        # Step 13: Find two groups based on bandwidth of selected sectors
        bandwidth_sectors = find_two_groups(bandwidth[selected_sector])
        bandwidth_sectors_filtered = list(
            filter(lambda x: len(x) > 0, bandwidth_sectors))
        # Step 14: Find positions in the bandwidth matrix for sector 1 and sector 2
        sector_1, sector_2 = find_positions(
            bandwidth, bandwidth_sectors_filtered)
        # Step 15: Write parameters for sector 1
        ds1, s1 = calculate_ds(bandwidth[sector_1] * 10)
        Kernel_outFile.write(str(sum(total_connectivity[sector_1])) + ',' + str(s1) + ',' + str(ds1) + ',' + str(
            np.mean(wgt_distance[sector_1])) + ',' + str(np.sum(effective_reefs[sector_1]) / np.sum(total_reefs[sector_1])) + ',')
        # Step 16: Write parameters for sector 2 if it exists
        if (len(sector_2) > 0):
            ds2, s2 = calculate_ds(bandwidth[sector_2] * 10)
            Kernel_outFile.write(str(sum(total_connectivity[sector_2])) + ',' + str(s2) + ',' + str(ds2) + ',' + str(
                np.mean(wgt_distance[sector_2])) + ',' + str(np.sum(effective_reefs[sector_2]) / np.sum(total_reefs[sector_2])) + '\n')
        else:
            Kernel_outFile.write('nan,nan,nan,nan,nan\n')
    Kernel_outFile.close()
