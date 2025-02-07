import math
import numpy as np
from .utils import veclength

def calculate_angle(a, b):
    """Calculate the angle between two vectors.
    
    Args:
        a (list/array): First 2D vector
        b (list/array): Second 2D vector
        
    Returns:
        float: Angle in degrees
    """
    dp = np.dot(a, b)
    la = veclength(a)
    lb = veclength(b)
    costheta = dp / (la * lb)
    rads = math.acos(costheta)
    return 180 * rads / math.pi

def calculate_direction_sector(angle):
    """Calculate the direction sector based on the angle.
    
    Args:
        angle (float): Angle in degrees
        
    Returns:
        int: Direction sector (0-35)
    """
    rot_angle = angle + 22.5
    return math.floor(rot_angle / 10) % 36

def haversine(coord1, coord2):
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


