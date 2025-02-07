import geopandas as gpd
import numpy as np
import math
import glob
from .utils import angle_circ_distance

def remove_outliers(connectivity, angles, centroids, distance, percent):
    # Remove the percent of locations with smallest and largest recruitment values
    num_outliers = int(len(np.where(connectivity)[0] > 0) * percent / 100)
    sorted_indices = np.argsort(connectivity)
    if (num_outliers == 0):
        filtered_indices = sorted_indices
    else:
        filtered_indices = sorted_indices[num_outliers:-num_outliers]
    filtered_connectivity = [connectivity[i] for i in filtered_indices]
    filtered_distance = [distance[i] for i in filtered_indices]
    filtered_angles = [angles[i] for i in filtered_indices]
    filtered_centroids = [centroids[i] for i in filtered_indices]
    return filtered_connectivity, filtered_angles, filtered_centroids, filtered_distance


def count_sectors(angle):
    # Count total recruitment values in sectors of 10 degrees bandwidth
    sectors = np.zeros(len(angle))
    for a in range(0, len(angle)):
        sectors[a] = int(angle[a] / 10)
    return sectors


def connectivity_by_sectors(sectors, connectivity, distance):
    # Calculate the total connectivity by sector
    sectors = np.array(sectors)
    connectivity = np.array(connectivity)
    distance = np.array(distance)
    vector = np.vectorize(np.int_)
    unique_sector = np.unique(sectors)
    sum_connectivity = []
    total_reefs = []
    eff_reefs = []
    avg_distance = []
    bandwidth = []
    weight_avg = []
    for band in unique_sector:
        index = np.array(np.argwhere(sectors == band))
        index = vector(index)
        total_reefs.append(len(index))
        sum_connectivity.append(connectivity[index].sum())
        if (np.sum(connectivity[index]) == 0):
            weights = np.zeros(len(index))
            eff_reefs.append(0)
        else:
            weights = (connectivity[index] /
                       np.sum(connectivity[index])).ravel()
            eff_reefs.append(np.count_nonzero(connectivity[index]))
        weight_avg.append(np.sum(distance[index].ravel() * weights))
        avg_distance.append(distance[index].mean())
        bandwidth.append(band)
    return np.array(sum_connectivity), np.array(bandwidth), np.array(avg_distance), np.array(weight_avg), np.array(eff_reefs), np.array(total_reefs)


def select_top_sectors(connectivity_array, num_sectors):
    # Select the num_sectors sectors with the highest connectivity value
    connectivity_array = np.array(connectivity_array)
    top_sectors = sorted(connectivity_array, reverse=True)[:num_sectors]
    top_sectors = np.trim_zeros(top_sectors)
    selected = np.where(np.isin(connectivity_array, top_sectors))[0]
    return np.array(selected)


def sort_angles(selected_bandwidth):
    """"
    This function arrange the vector of angles by distance between each other
    """
    selected_bandwidth = selected_bandwidth * 10
    sorted_angle_array = np.zeros(len(selected_bandwidth))
    sorted_angle_array[0] = selected_bandwidth[0]
    selected_bandwidth = np.delete(selected_bandwidth, 0)
    current = 0
    effective_distance = []
    while (len(selected_bandwidth) > 0):
        distances = []
        for i in range(0, len(selected_bandwidth)):
            distances.append(angle_circ_distance(
                sorted_angle_array[current], selected_bandwidth[i]))
        next_in_line = np.argmin(distances)
        effective_distance.append(np.min(distances))
        current += 1
        sorted_angle_array[current] = selected_bandwidth[next_in_line]
        selected_bandwidth = np.delete(selected_bandwidth, next_in_line)
    sorted_angle_array = [int(angle/10) for angle in sorted_angle_array]
    return (np.array(sorted_angle_array), np.array(effective_distance))


def select_boundary_sectors(top_sectors, total_connectivity):
    """"
    Select the boundary sectors around the top sectors
    """
    selected = np.array(
        np.unique(np.array((top_sectors + 1, top_sectors - 1)).ravel()))
    # Remove duplicated and negative index
    if (any(selected < 0)):
        selected[np.where(selected < 0)] = len(total_connectivity)-1
    if (any(selected >= len(total_connectivity))):
        selected[np.where(selected >= len(total_connectivity))] = 0
    selected = np.unique(selected)
    selected = selected[~np.in1d(selected, top_sectors)]
    temp_connectivity = sorted(
        total_connectivity[np.array(selected)], reverse=True)[0]
    final_select = np.array(
        np.where((temp_connectivity == total_connectivity)))
    return (final_select)


def find_two_groups(sectors_array):
    """
    Finds two groups of nearest values in a given list.
    """
    angles, distances = sort_angles(sectors_array)
    angle_sector_diff = np.diff(angles)
    # if the distance is higher than 1 (which means 10 degrees), it should find 2 groups.
    if (len(angles) > 1 and max(angle_sector_diff) > 1):
        cutting = int(np.max(np.where(distances == distances.max()))) + 1
        return angles[:cutting], angles[cutting:]
    else:
        return angles, []


def find_positions(list_of_sectors, sectors):
    """
    Finds the positions of specific values within a list.
    """
    if (len(sectors) > 1):
        first_sector = np.where(np.in1d(list_of_sectors, sectors[0]))[0]
        second_sector = np.where(np.in1d(list_of_sectors, sectors[1]))[0]
        return first_sector, second_sector
    else:
        only_sector = np.where(np.in1d(list_of_sectors, sectors[0]))[0]
        return only_sector, []


def calculate_ds(vector_angles):
    """
    Find the total angle from the source (DS) and the distance from the angle 0 (S)
    """
    DS = np.max(vector_angles) - np.min(vector_angles)
    if (DS == 0):
        DS = 10
    S = np.min(vector_angles) + (DS / 2)
    if (np.min(vector_angles) < np.max(vector_angles) - 180):
        DS = 360 - np.max(vector_angles) + np.min(vector_angles)
        S = (np.max(vector_angles) + (DS / 2)) % 360
    return (DS, S)

