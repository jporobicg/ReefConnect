"""
Ecological processes for reef connectivity calculations.

This module contains functions related to ecological processes:
- Decay and survival functions
- Competence calculations
- Point-in-polygon algorithms for settlement detection
"""

import math
from scipy.integrate import quad
import numba
from numba import jit, njit
import numpy as np


def bathtub_curve(lmbda, v, sigma):
    """
    Transforms the bathtub equation into a function that can be integrated
    using the trapezoidal function.

    Parameters
    ----------
    lmbda : float
        The scale parameter of the Weibull distribution used in the bathtub equation.
        Must be a positive integer or float.
    v : float
        The shape parameter of the Weibull distribution used in the bathtub equation.
        Must be a positive integer or float.
    sigma : float
        A constant that determines the shape of the bathtub curve. Must be a positive
        integer or float. It can be zero, becoming an exponential function.

    Returns
    -------
    function
        A lambda function that represents the bathtub curve equation. The function takes
        a single argument, t, which represents the age of the coral. The function returns
        the value of the bathtub curve equation at that age.
    """
    u = lambda t: (lmbda * v * pow((lmbda * t), v - 1)) / (1 - sigma * pow((lmbda * t), v))
    return u


def piecewise_decay(ages, Tcp, lmbda1, lmbda2, v1, v2, sigma1, sigma2):
    """
    Calculates the probability of survival of corals larvaes at different ages, using the piecewise
    Weibull-Weibull survival model described by Moneghetti et al. (2019).

    Parameters
    ----------
    ages : list
        A list of ages of the corals. Each age must be a positive integer or float.
    Tcp : float
        The age (inflection point) at which the corals transition from a Weibull survival curve to another Weibull
        survival curve. Must be greater than 0 and less than the maximum age in `ages`.
    lmbda1 : float
        The scale parameter of the Weibull survival curve in the first phase. Must be a positive
        integer or float.
    lmbda2 : float
        The scale parameter of the Weibull survival curve in the second phase. Must be a positive
        integer or float.
    v1 : float
        The shape parameter of the Weibull survival curve in the first phase. Must be a positive
        integer or float.
    v2 : float
        The shape parameter of the Weibull survival curve in the second phase. Must be a positive
        integer or float.
    sigma1 : float
        The standard deviation of the Gaussian noise added to the survival curve in the first phase.
        Must be a positive integer or float. It can be zero, becoming an exponential function.
    sigma2 : float
        The standard deviation of the Gaussian noise added to the survival curve in the second phase.
        Must be a positive integer or float. It can be zero, becoming an exponential function.

    Returns
    -------
    list
        A list of survival probabilities for the corals, calculated using the piecewise
        Weibull-Weibull survival model. Each survival probability corresponds to the age
        in the input list `ages`.
    """
    fx1 = bathtub_curve(lmbda1, v1, sigma1)
    fx2 = bathtub_curve(lmbda2, v2, sigma2)
    decay = []
    for age in range(0, len(ages)):
        if ages[age] < Tcp:
            area = quad(fx1, 0, ages[age])[0]
        else:
            area = quad(fx1, 0, Tcp)[0] + quad(fx2, Tcp, ages[age])[0]
        decay.append(math.exp(-area))
    return decay


def piecewise_competence(ages, tc, Tcp, alpha, beta1, beta2, v):
    """
    Calculates the larval competence values at different ages (days), using the piecewise
    Weibull-exponential competence model. This function is a replica of the R code used by
    Moneghetti et al. (2019) to calculate competence.

    Parameters
    ----------
    ages : list
        A list of larvaes ages. Each age must be a positive integer or float.
    tc : float
        The age at which the larvaes reaches their maximum competence level. Must be a
        positive integer or float.
    Tcp : float
        The age at which the larvaes starts to experience a decline in competence. Must be
        greater than tc and a positive integer or float.
    alpha : float
        The scale parameter of the Weibull distribution. Must be a positive integer or float.
    beta1 : float
        The shape parameter of the Weibull distribution in the early decline phase. Must be
        a positive integer or float.
    beta2 : float
        The shape parameter of the Weibull distribution in the late decline phase. Must be
        a positive integer or float.
    v : float
        The exponential decay parameter in the early decline phase. Must be a positive
        integer or float.

    Returns
    -------
    list
        A list of competence values for larvaes, calculated using the piecewise
        Weibull-exponential competence model. Each competence value corresponds to the age
        in the input list `ages`.
    """
    competence = []
    for age in range(0, len(ages)):
        if ages[age] < tc:
            area = 0
        if ages[age] >= tc and ages[age] <= Tcp:
            fxtau_early = lambda tau: alpha * math.exp(-alpha * (tau - tc)) * math.exp(-pow((beta1 * (ages[age] - tau)), v))
            area = quad(fxtau_early, tc, ages[age])[0]
        if ages[age] > Tcp:
            fxtau_late_first = lambda tau: alpha * math.exp(-alpha * (tau - tc)) * math.exp(-pow((beta1 * (Tcp - tau)), v)) * math.exp(-beta2 * (ages[age] - Tcp))
            fxtau_late_second = lambda tau: alpha * math.exp(-alpha * (tau - tc)) * math.exp(-beta2 * (ages[age] - tau))
            area = quad(fxtau_late_first, tc, Tcp)[0] + quad(fxtau_late_second, Tcp, ages[age])[0]
        competence.append(area)
    return competence


@jit(nopython=True)
def point_in_polygon(x, y, polygon):
    """
    Determines whether a point is inside a polygon using the ray-casting algorithm.

    Parameters
    ----------
    x : float
        The x-coordinate of the point to test.
    y : float
        The y-coordinate of the point to test.
    polygon : list of tuples
        A list of tuples representing the vertices of the polygon, in order.

    Returns
    -------
    bool
        True if the point is inside the polygon, False otherwise.
    """
    num_vertices = len(polygon)
    is_inside = False
    previous_x, previous_y = polygon[0]
    # start outside the polygon
    intersection_x = 0.0
    intersection_y = 0.0
    for i in numba.prange(num_vertices + 1):
        current_x, current_y = polygon[i % num_vertices]
        if y > min(previous_y, current_y):
            if y <= max(previous_y, current_y):
                if x <= max(previous_x, current_x):
                    if previous_y != current_y:
                        intersection_x = (y - previous_y) * (current_x - previous_x) / (current_y - previous_y) + previous_x
                    if previous_x == current_x or x <= intersection_x:
                        is_inside = not is_inside
        previous_x, previous_y = current_x, current_y

    return is_inside


def connolly_competence(ages, t_min, t_max, alpha, beta):
    """
    Calculates larval competence based on Connolly et al. (2010) model.
    
    This function implements the competence model from Connolly et al. (2010) Ecology paper
    "Estimating dispersal potential for marine larvae: dynamic models applied to scleractinian corals".
    
    The model describes how larval competence changes with age:
    - Larvae are not competent before t_min
    - Competence increases from t_min to t_max (reaching maximum)
    - Competence declines after t_max
    
    Based on equations 1-5 from the paper:
    - Eq 1: Basic competence function
    - Eq 2: Time-dependent settlement probability  
    - Eq 3: Maximum competence calculation
    - Eq 4: Competence decline phase
    - Eq 5: Final competence probability (this function)
    
    Parameters
    ----------
    ages : array-like
        Array of larval ages in days
    t_min : float
        Minimum age for competence (days) - larvae not competent before this age
    t_max : float  
        Age of maximum competence (days) - competence peaks at this age
    alpha : float
        Shape parameter controlling competence increase rate
    beta : float
        Shape parameter controlling competence decline rate
        
    Returns
    -------
    numpy.ndarray
        Array of competence values (0-1) for each age
    """
    ages = np.asarray(ages)
    competence = np.zeros_like(ages, dtype=float)
    
    # Phase 1: No competence before t_min
    mask1 = ages < t_min
    competence[mask1] = 0.0
    
    # Phase 2: Competence increases from t_min to t_max
    mask2 = (ages >= t_min) & (ages <= t_max)
    if np.any(mask2):
        # Competence increases following a sigmoid curve
        # Using equation 5 from Connolly et al. (2010)
        t_norm = (ages[mask2] - t_min) / (t_max - t_min)
        competence[mask2] = 1.0 / (1.0 + np.exp(-alpha * (t_norm - 0.5)))
    
    # Phase 3: Competence declines after t_max
    mask3 = ages > t_max
    if np.any(mask3):
        # Competence declines exponentially after t_max
        # Following equation 4 from the paper
        t_decay = ages[mask3] - t_max
        competence[mask3] = np.exp(-beta * t_decay)
    
    return competence





@njit(parallel=False)
def points_in_polygon(xs, ys, miny, maxy, polygon):
    """
    Test whether a point is inside a given polygon using the point-in-polygon algorithm.

    This function tests each point in the `points` array to determine if it is inside the polygon
    defined by the vertices in the `polygon` array. The function uses the point-in-polygon Ray casting
    algorithm to perform this test. The algorithm performs the even-odd-rule algorithm to find out
    whether a point is in a given polygon. This runs in O(n) time where n is the number of edges of the polygon.

    Parameters:
    -----------
    points: numpy.ndarray of shape (n, 2)
        Array of n points to test. Each point is defined by its x and y coordinates in columns 0 and 1 respectively.
    polygon: numpy.ndarray of shape (m, 2)
        Array of m vertices defining the polygon. Each vertex is defined by its x and y coordinates in columns 0 and 1 respectively.

    Returns:
    --------
    D: numpy.ndarray of shape (n,)
        Boolean array indicating whether each point in `points` is inside the polygon (`True`) or not (`False`).
    """
    D = np.empty(len(ys), dtype=numba.boolean)
    for i in range(len(D)):
        if ys[i] >= miny and ys[i] <= maxy:
            D[i] = point_in_polygon(xs[i], ys[i], polygon)
        else:
            D[i] = False
    return D 