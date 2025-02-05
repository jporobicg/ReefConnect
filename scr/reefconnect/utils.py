import math
import numpy as np

def veclength(vector):
    """Calculate the size (length) of a 2D vector.
    
    Args:
        vector (list/array): 2D vector with x,y components
        
    Returns:
        float: Length of the vector
    """
    return math.sqrt(math.pow(vector[0], 2) + math.pow(vector[1], 2))

def haversine(coord1, coord2):
    """Calculate the great-circle distance between two points on Earth.
    
    Args:
        coord1 (tuple): (latitude, longitude) of first point
        coord2 (tuple): (latitude, longitude) of second point
        
    Returns:
        float: Distance in kilometers
    """
    radius = 6371  # Earth's radius in km
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c