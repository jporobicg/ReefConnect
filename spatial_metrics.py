"""
Spatial metrics for reef connectivity calculations.

This module contains functions for calculating spatial relationships between reefs:
- Vector operations and angle calculations
- Distance calculations using Haversine formula
- Directional calculations
"""

import numpy as np
import math


def veclength(vector):
    """
    Calculate the length (magnitude) of a 2D vector.
    
    Parameters
    ----------
    vector : list or array
        Vector of size 2 [x, y].
        
    Returns
    -------
    float
        Length of the vector.
    """
    value = math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))
    return value


def angle(a, b):
    """
    Calculate the angle between two vectors.
    
    Parameters
    ----------
    a : list or array
        First vector [x, y].
    b : list or array
        Second vector [x, y].
        
    Returns
    -------
    float
        Angle in degrees between the two vectors.
    """
    dp = np.dot(a, b)  # Dot product
    la = veclength(a)
    lb = veclength(b)
    costheta = dp / (la * lb)
    rads = math.acos(costheta)
    angle_val = 180 * rads / math.pi
    return angle_val


def haversine(coord1, coord2):
    """
    Calculate the great circle distance between two points on Earth.
    
    Parameters
    ----------
    coord1 : tuple
        First coordinate (lat, lon) in degrees.
    coord2 : tuple
        Second coordinate (lat, lon) in degrees.
        
    Returns
    -------
    float
        Distance in kilometers.
    """
    # Earth's radius in km
    radius = 6371
    # Convert coordinates to radians
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    # Calculate differences between coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Calculate Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Calculate distance in km
    distance = radius * c
    return distance


def calculate_angles_and_distances(reef_centroids, num_sites):
    """
    Calculate angles and distances between all reef pairs.
    
    Parameters
    ----------
    reef_centroids : list
        List of reef centroids.
    num_sites : int
        Number of reef sites.
        
    Returns
    -------
    tuple
        Angle matrix, direction matrix, and distance matrix.
    """
    # Initialize matrices
    angle_matrix = np.zeros((num_sites, num_sites))
    distance_matrix = np.zeros((num_sites, num_sites))
    direction_matrix = np.zeros((num_sites, num_sites))
    
    # Reference vector along Y axis (angle 0)
    a = [0, 1]
    
    for release_site in range(num_sites):
        for target_site in range(num_sites):
            if release_site != target_site:
                # Calculate coordinates (using original approach)
                target_centroid = reef_centroids[target_site]
                source_centroid = reef_centroids[release_site]
                
                coordinates_sink = np.array([target_centroid.x, target_centroid.y])
                coordinates_source = np.array([source_centroid.x, source_centroid.y])
                
                # Calculate vector between reefs
                b = [coordinates_sink[0] - coordinates_source[0], coordinates_sink[1] - coordinates_source[1]]
                
                # Calculate angle
                reef_angle = angle(a, b)
                if b[0] < 0:
                    reef_angle = 360 - reef_angle
                
                # Add rotation and calculate direction sector
                rot_reef_angle = reef_angle + 22.5
                direction = math.floor(rot_reef_angle / 10) % 36
                
                # Calculate distance
                distance = haversine(coordinates_source, coordinates_sink)
                
                angle_matrix[release_site, target_site] = reef_angle
                direction_matrix[release_site, target_site] = direction
                distance_matrix[release_site, target_site] = distance
    
    return angle_matrix, direction_matrix, distance_matrix 