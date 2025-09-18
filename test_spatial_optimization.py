#!/usr/bin/env python3
"""
Test script for spatial filtering optimization.
Compares old point-in-polygon approach vs new geopandas spatial join.
"""

import time
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
from io_utils import fast_spatial_filtering, quick_bounding_box_filter

def create_test_data(n_particles=1000, n_reefs=10):
    """
    Create test data for spatial filtering optimization.
    
    Parameters
    ----------
    n_particles : int
        Number of test particles
    n_reefs : int
        Number of test reefs
        
    Returns
    -------
    tuple
        (particles_df, data_shape, candidate_reefs)
    """
    print(f"Creating test data: {n_particles} particles, {n_reefs} reefs")
    
    # Create random particles in a bounded area
    np.random.seed(42)  # For reproducible results
    lons = np.random.uniform(145, 155, n_particles)  # GBR longitude range
    lats = np.random.uniform(-25, -10, n_particles)  # GBR latitude range
    ages = np.random.uniform(1, 30, n_particles)
    trajs = np.random.randint(0, 500, n_particles)
    
    particles_df = pd.DataFrame({
        'longitudes': lons,
        'latitudes': lats,
        'age': ages,
        'trajectories': trajs
    })
    
    # Create test reefs (rectangular polygons)
    reef_geometries = []
    reef_ids = []
    
    for i in range(n_reefs):
        # Create rectangular reef
        center_lon = np.random.uniform(146, 154)
        center_lat = np.random.uniform(-24, -11)
        width = 0.1  # degrees
        height = 0.1  # degrees
        
        reef_polygon = Polygon([
            (center_lon - width/2, center_lat - height/2),
            (center_lon + width/2, center_lat - height/2),
            (center_lon + width/2, center_lat + height/2),
            (center_lon - width/2, center_lat + height/2),
            (center_lon - width/2, center_lat - height/2)
        ])
        
        reef_geometries.append(reef_polygon)
        reef_ids.append(i)
    
    # Create GeoDataFrame for reefs
    data_shape = gpd.GeoDataFrame({
        'reef_id': reef_ids,
        'geometry': reef_geometries
    }, crs='EPSG:4326')
    
    # All reefs are candidates for testing
    candidate_reefs = list(range(n_reefs))
    
    return particles_df, data_shape, candidate_reefs

def old_spatial_filtering(particles_df, candidate_reefs, data_shape):
    """
    Old slow approach: individual point-in-polygon testing.
    """
    settled_particles_by_reef = {}
    
    for sink_reef_id in candidate_reefs:
        sink_polygon = data_shape['geometry'][sink_reef_id]
        
        # Find particles in this reef using slow approach
        particle_points = [Point(lon, lat) for lon, lat in zip(particles_df['longitudes'], particles_df['latitudes'])]
        in_polygon_mask = [sink_polygon.contains(point) for point in particle_points]
        settled_indices = np.where(in_polygon_mask)[0]
        
        if len(settled_indices) > 0:
            settled_particles_by_reef[sink_reef_id] = settled_indices.tolist()
    
    return settled_particles_by_reef

def test_spatial_filtering_correctness():
    """
    Test that old and new approaches give identical results.
    """
    print("="*60)
    print("TESTING SPATIAL FILTERING CORRECTNESS")
    print("="*60)
    
    # Create test data
    particles_df, data_shape, candidate_reefs = create_test_data(n_particles=100, n_reefs=5)
    
    # Test old approach
    print("Running old approach...")
    old_results = old_spatial_filtering(particles_df, candidate_reefs, data_shape)
    
    # Test new approach
    print("Running new approach...")
    new_results = fast_spatial_filtering(particles_df, candidate_reefs, data_shape)
    
    # Compare results
    print("\nComparing results...")
    print(f"Old approach found {len(old_results)} reefs with particles")
    print(f"New approach found {len(new_results)} reefs with particles")
    
    # Check if results match
    all_match = True
    for reef_id in candidate_reefs:
        old_set = set(old_results.get(reef_id, []))
        new_set = set(new_results.get(reef_id, []))
        
        if old_set != new_set:
            print(f"❌ Mismatch for reef {reef_id}:")
            print(f"   Old: {sorted(old_set)}")
            print(f"   New: {sorted(new_set)}")
            all_match = False
    
    if all_match:
        print("✅ CORRECTNESS TEST PASSED: Results match exactly!")
    else:
        print("❌ CORRECTNESS TEST FAILED: Results don't match!")
    
    return all_match

def test_spatial_filtering_performance():
    """
    Test performance improvement of new approach.
    """
    print("\n" + "="*60)
    print("TESTING SPATIAL FILTERING PERFORMANCE")
    print("="*60)
    
    # Test with different data sizes
    test_sizes = [
        (100, 5),    # Small
        (500, 10),   # Medium
        (1000, 20),  # Large
        (2000, 50),  # Very large
    ]
    
    results = []
    
    for n_particles, n_reefs in test_sizes:
        print(f"\nTesting with {n_particles} particles, {n_reefs} reefs...")
        
        # Create test data
        particles_df, data_shape, candidate_reefs = create_test_data(n_particles, n_reefs)
        
        # Test old approach
        start_time = time.time()
        old_results = old_spatial_filtering(particles_df, candidate_reefs, data_shape)
        old_time = time.time() - start_time
        
        # Test new approach
        start_time = time.time()
        new_results = fast_spatial_filtering(particles_df, candidate_reefs, data_shape)
        new_time = time.time() - start_time
        
        # Calculate speedup
        speedup = old_time / new_time if new_time > 0 else float('inf')
        
        print(f"   Old approach: {old_time:.4f} seconds")
        print(f"   New approach: {new_time:.4f} seconds")
        print(f"   Speedup: {speedup:.1f}x")
        
        results.append({
            'n_particles': n_particles,
            'n_reefs': n_reefs,
            'old_time': old_time,
            'new_time': new_time,
            'speedup': speedup
        })
    
    return results

def test_bounding_box_filter():
    """
    Test the bounding box pre-filter optimization.
    """
    print("\n" + "="*60)
    print("TESTING BOUNDING BOX PRE-FILTER")
    print("="*60)
    
    # Create test data
    particles_df, data_shape, candidate_reefs = create_test_data(n_particles=500, n_reefs=20)
    
    # Test bounding box filter
    start_time = time.time()
    filtered_reefs = quick_bounding_box_filter(particles_df, candidate_reefs, data_shape)
    filter_time = time.time() - start_time
    
    print(f"Original candidate reefs: {len(candidate_reefs)}")
    print(f"Filtered candidate reefs: {len(filtered_reefs)}")
    print(f"Filtering time: {filter_time:.4f} seconds")
    print(f"Reduction: {len(candidate_reefs) - len(filtered_reefs)} reefs eliminated")
    
    return filtered_reefs

def plot_performance_results(results):
    """
    Plot performance comparison results.
    """
    print("\n" + "="*60)
    print("CREATING PERFORMANCE PLOTS")
    print("="*60)
    
    # Extract data for plotting
    n_particles = [r['n_particles'] for r in results]
    old_times = [r['old_time'] for r in results]
    new_times = [r['new_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Execution time comparison
    ax1.plot(n_particles, old_times, 'r-o', label='Old Approach', linewidth=2)
    ax1.plot(n_particles, new_times, 'b-s', label='New Approach', linewidth=2)
    ax1.set_xlabel('Number of Particles')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Spatial Filtering Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Speedup
    ax2.plot(n_particles, speedups, 'g-^', label='Speedup', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Particles')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Performance Improvement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spatial_optimization_test.png', dpi=150, bbox_inches='tight')
    print("✅ Performance plots saved as 'spatial_optimization_test.png'")
    
    return fig

def main():
    """
    Run all spatial filtering optimization tests.
    """
    print("🚀 SPATIAL FILTERING OPTIMIZATION TEST")
    print("="*60)
    
    # Test 1: Correctness
    correctness_passed = test_spatial_filtering_correctness()
    
    if not correctness_passed:
        print("\n❌ Correctness test failed! Stopping here.")
        return False
    
    # Test 2: Performance
    performance_results = test_spatial_filtering_performance()
    
    # Test 3: Bounding box filter
    filtered_reefs = test_bounding_box_filter()
    
    # Test 4: Create plots
    if HAS_MATPLOTLIB:
        plot_performance_results(performance_results)
    else:
        print("⚠️  Matplotlib not available, skipping plots")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Correctness: {'PASSED' if correctness_passed else 'FAILED'}")
    print(f"✅ Performance: Tested with {len(performance_results)} different sizes")
    print(f"✅ Bounding box filter: {len(filtered_reefs)} reefs after filtering")
    
    avg_speedup = np.mean([r['speedup'] for r in performance_results])
    print(f"✅ Average speedup: {avg_speedup:.1f}x")
    
    if avg_speedup > 5:
        print("🎉 OPTIMIZATION SUCCESSFUL! Ready for production use.")
    else:
        print("⚠️  Optimization needs improvement.")
    
    return True

if __name__ == "__main__":
    main()
