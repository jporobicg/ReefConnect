#!/usr/bin/env python3
"""
Print Connectivity Summary
=========================

Print a text summary of the connectivity results for the processed reefs.
"""

import xarray as xr
import numpy as np


def print_connectivity_summary(nc_file, reef_ids=[0, 1, 2, 3]):
    """
    Print a detailed text summary of connectivity results.
    """
    print("="*80)
    print("CONNECTIVITY RESULTS SUMMARY")
    print("="*80)
    
    # Load the data
    ds = xr.open_dataset(nc_file)
    reef_subset = ds.sel(source=reef_ids, sink=reef_ids)
    
    print(f"Analysis of {len(reef_ids)} reefs: {reef_ids}")
    print(f"Bootstrap samples: {ds.dims['sample']}")
    print()
    
    # Print detailed statistics for each reef
    for reef_id in reef_ids:
        print(f"REEF {reef_id} ANALYSIS")
        print("-" * 40)
        
        # Self-connectivity
        self_conn = reef_subset['connectivity'].sel(source=reef_id, sink=reef_id)
        self_mean = self_conn.mean(dim='sample').values
        self_std = self_conn.std(dim='sample').values
        self_min = self_conn.min(dim='sample').values
        self_max = self_conn.max(dim='sample').values
        
        print(f"Self-connectivity:")
        print(f"  Mean ± Std: {self_mean:.6f} ± {self_std:.6f}")
        print(f"  Range: {self_min:.6f} - {self_max:.6f}")
        print()
        
        # Outgoing connectivity to other reefs
        print("Outgoing connectivity to other reefs:")
        for sink_id in reef_ids:
            if reef_id != sink_id:
                conn = reef_subset['connectivity'].sel(source=reef_id, sink=sink_id)
                dist = reef_subset['distance'].sel(source=reef_id, sink=sink_id).values
                mean_conn = conn.mean(dim='sample').values
                std_conn = conn.std(dim='sample').values
                
                print(f"  Reef {reef_id} → Reef {sink_id}:")
                print(f"    Distance: {dist:.2f} km")
                print(f"    Connectivity: {mean_conn:.6f} ± {std_conn:.6f}")
        
        print()
        
        # Incoming connectivity from other reefs
        print("Incoming connectivity from other reefs:")
        for source_id in reef_ids:
            if reef_id != source_id:
                conn = reef_subset['connectivity'].sel(source=source_id, sink=reef_id)
                dist = reef_subset['distance'].sel(source=source_id, sink=reef_id).values
                mean_conn = conn.mean(dim='sample').values
                std_conn = conn.std(dim='sample').values
                
                print(f"  Reef {source_id} → Reef {reef_id}:")
                print(f"    Distance: {dist:.2f} km")
                print(f"    Connectivity: {mean_conn:.6f} ± {std_conn:.6f}")
        
        print()
        print("="*80)
        print()
    
    # Print pairwise comparison
    print("PAIRWISE CONNECTIVITY COMPARISON")
    print("-" * 40)
    
    for i, reef1 in enumerate(reef_ids):
        for j, reef2 in enumerate(reef_ids):
            if i < j:  # Only print each pair once
                conn_1to2 = reef_subset['connectivity'].sel(source=reef1, sink=reef2)
                conn_2to1 = reef_subset['connectivity'].sel(source=reef2, sink=reef1)
                dist = reef_subset['distance'].sel(source=reef1, sink=reef2).values
                
                mean_1to2 = conn_1to2.mean(dim='sample').values
                mean_2to1 = conn_2to1.mean(dim='sample').values
                
                print(f"Reef {reef1} ↔ Reef {reef2} (Distance: {dist:.2f} km):")
                print(f"  {reef1} → {reef2}: {mean_1to2:.6f}")
                print(f"  {reef2} → {reef1}: {mean_2to1:.6f}")
                print(f"  Asymmetry: {abs(mean_1to2 - mean_2to1):.6f}")
                print()
    
    # Print overall statistics
    print("OVERALL STATISTICS")
    print("-" * 40)
    
    all_connectivity = reef_subset['connectivity'].values
    all_distances = reef_subset['distance'].values
    
    print(f"Total connectivity values: {all_connectivity.size}")
    print(f"Non-zero connectivity values: {np.count_nonzero(all_connectivity)}")
    print(f"Connectivity range: {np.min(all_connectivity):.6f} - {np.max(all_connectivity):.6f}")
    print(f"Mean connectivity: {np.mean(all_connectivity):.6f}")
    print(f"Std connectivity: {np.std(all_connectivity):.6f}")
    print()
    print(f"Distance range: {np.min(all_distances):.2f} - {np.max(all_distances):.2f} km")
    print(f"Mean distance: {np.mean(all_distances):.2f} km")
    print()
    
    # Check for any issues
    print("DATA QUALITY CHECK")
    print("-" * 40)
    
    nan_count = np.isnan(all_connectivity).sum()
    inf_count = np.isinf(all_connectivity).sum()
    neg_count = (all_connectivity < 0).sum()
    
    print(f"NaN values: {nan_count}")
    print(f"Infinite values: {inf_count}")
    print(f"Negative values: {neg_count}")
    
    if nan_count == 0 and inf_count == 0 and neg_count == 0:
        print("✅ Data quality: Excellent")
    else:
        print("⚠️  Data quality: Issues detected")
    
    ds.close()


def main():
    """Main function."""
    nc_file = "output/connectivity_results_example.nc"
    
    try:
        print_connectivity_summary(nc_file)
        print("\n" + "="*80)
        print("SUMMARY COMPLETE")
        print("="*80)
        print("✅ Text summary has been printed above")
        print("📊 Plots have been saved to output/ directory:")
        print("   - connectivity_plots.png (overview)")
        print("   - reef_X_analysis.png (individual reefs)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main() 