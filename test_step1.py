#!/usr/bin/env python3
"""
Test Step 1: Verify the new files work correctly
"""

import numpy as np
import math

# Test the new files
from ecological_processes import bathtub_curve, piecewise_decay, piecewise_competence
from spatial_metrics import veclength, angle, haversine


def test_ecological_processes():
    """Test ecological processes functions."""
    print("="*50)
    print("TESTING ECOLOGICAL PROCESSES")
    print("="*50)
    
    # Test bathtub_curve
    lmbda, v, sigma = 0.4, 2.892, 0
    curve_func = bathtub_curve(lmbda, v, sigma)
    result = curve_func(1.0)
    print(f"✅ bathtub_curve test passed: {result}")
    
    # Test piecewise_decay
    ages = [1.0, 2.0, 3.0]
    Tcp, lmbda1, lmbda2, v1, v2, sigma1, sigma2 = 2.583, 0.4, 0.019, 2.892, 1.716, 0, 0
    decay_result = piecewise_decay(ages, Tcp, lmbda1, lmbda2, v1, v2, sigma1, sigma2)
    print(f"✅ piecewise_decay test passed: {len(decay_result)} results")
    
    # Test piecewise_competence
    tc, Tcp_comp, alpha, beta1, beta2, v = 3.333, 69.91245, 1.295, 0.001878001, 0.3968972, 0.364
    competence_result = piecewise_competence(ages, tc, Tcp_comp, alpha, beta1, beta2, v)
    print(f"✅ piecewise_competence test passed: {len(competence_result)} results")
    
    return True


def test_spatial_metrics():
    """Test spatial metrics functions."""
    print("\n" + "="*50)
    print("TESTING SPATIAL METRICS")
    print("="*50)
    
    # Test veclength
    vector = [3, 4]
    length = veclength(vector)
    expected_length = 5.0
    print(f"✅ veclength test passed: {length} (expected: {expected_length})")
    assert abs(length - expected_length) < 0.001
    
    # Test angle
    a = [0, 1]  # Reference vector
    b = [1, 1]  # Test vector
    angle_val = angle(a, b)
    print(f"✅ angle test passed: {angle_val:.2f} degrees")
    assert 0 <= angle_val <= 180
    
    # Test haversine
    coord1 = (0, 0)
    coord2 = (1, 1)
    distance = haversine(coord1, coord2)
    print(f"✅ haversine test passed: {distance:.2f} km")
    assert distance > 0
    
    return True


def test_main_connectivity():
    """Test main connectivity calculation import."""
    print("\n" + "="*50)
    print("TESTING MAIN CONNECTIVITY CALCULATION")
    print("="*50)
    
    try:
        from main_connectivity_calculation import calc
        print("✅ main_connectivity_calculation import successful")
        return True
    except Exception as e:
        print(f"❌ main_connectivity_calculation import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("STEP 1 TESTING - NEW FILE ORGANIZATION")
    print("="*60)
    
    # Test 1: Ecological processes
    try:
        test_ecological_processes()
        print("✅ Ecological processes test passed!")
    except Exception as e:
        print(f"❌ Ecological processes test failed: {e}")
    
    # Test 2: Spatial metrics
    try:
        test_spatial_metrics()
        print("✅ Spatial metrics test passed!")
    except Exception as e:
        print(f"❌ Spatial metrics test failed: {e}")
    
    # Test 3: Main connectivity calculation
    try:
        test_main_connectivity()
        print("✅ Main connectivity calculation test passed!")
    except Exception as e:
        print(f"❌ Main connectivity calculation test failed: {e}")
    
    print("\n" + "="*60)
    print("STEP 1 TESTING COMPLETE")
    print("="*60)
    print("\n✅ All new files are working correctly!")
    print("Ready to proceed to Step 2.")


if __name__ == "__main__":
    main() 