#!/usr/bin/env python3
"""
Compare CSV connectivity matrix with NetCDF test output
Focus on Moneghetti treatment from first sample
"""

import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

def load_csv_data(csv_file):
    """Load and analyze the CSV connectivity data"""
    print(f"\n=== LOADING CSV DATA: {csv_file} ===")
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return None
    
    # Load CSV
    df = pd.read_csv(csv_file)
    print(f"CSV shape: {df.shape}")
    print(f"Column names (first 5): {list(df.columns[:5])}")
    print(f"Column names (last 5): {list(df.columns[-5:])}")
    
    # Convert to numpy array
    matrix = df.values
    print(f"Matrix shape: {matrix.shape}")
    print(f"Data type: {matrix.dtype}")
    
    # Basic statistics
    print(f"Non-zero elements: {np.count_nonzero(matrix)}")
    print(f"Value range: {np.min(matrix):.2e} to {np.max(matrix):.2e}")
    print(f"Mean value: {np.mean(matrix):.2e}")
    print(f"Sparsity: {(1 - np.count_nonzero(matrix)/matrix.size)*100:.2f}%")
    
    return matrix

def load_netcdf_data(nc_file):
    """Load and analyze the NetCDF connectivity data (Moneghetti, first sample)"""
    print(f"\n=== LOADING NETCDF DATA: {nc_file} ===")
    
    if not os.path.exists(nc_file):
        print(f"❌ NetCDF file not found: {nc_file}")
        return None
    
    # Load NetCDF
    ds = xr.open_dataset(nc_file)
    print(f"NetCDF dimensions: {dict(ds.sizes)}")
    print(f"Treatments: {list(ds.treatment.values)}")
    
    # Get Moneghetti treatment, first sample
    conn_data = ds.connectivity.sel(treatment='moneghetti', sample=0)
    matrix = conn_data.values
    print(f"Matrix shape: {matrix.shape}")
    print(f"Data type: {matrix.dtype}")
    
    # Basic statistics
    print(f"Non-zero elements: {np.count_nonzero(matrix)}")
    print(f"Value range: {np.min(matrix):.2e} to {np.max(matrix):.2e}")
    print(f"Mean value: {np.mean(matrix):.2e}")
    print(f"Sparsity: {(1 - np.count_nonzero(matrix)/matrix.size)*100:.2f}%")
    
    return ds, matrix

def compare_matrices(csv_matrix, nc_matrix):
    """Compare the two matrices"""
    print(f"\n=== COMPARING MATRICES ===")
    
    print(f"CSV MATRIX:")
    print(f"  Shape: {csv_matrix.shape}")
    print(f"  Non-zero: {np.count_nonzero(csv_matrix)}")
    print(f"  Range: {np.min(csv_matrix):.2e} to {np.max(csv_matrix):.2e}")
    print(f"  Mean: {np.mean(csv_matrix):.2e}")
    
    print(f"NETCDF MATRIX (Moneghetti, sample 0):")
    print(f"  Shape: {nc_matrix.shape}")
    print(f"  Non-zero: {np.count_nonzero(nc_matrix)}")
    print(f"  Range: {np.min(nc_matrix):.2e} to {np.max(nc_matrix):.2e}")
    print(f"  Mean: {np.mean(nc_matrix):.2e}")
    
    # Check if we can compare directly
    if csv_matrix.shape == nc_matrix.shape:
        print(f"\n✅ Same shape - can compare directly")
        
        # Calculate differences
        diff = nc_matrix - csv_matrix
        abs_diff = np.abs(diff)
        
        print(f"DIFFERENCE ANALYSIS:")
        print(f"  Mean difference: {np.mean(diff):.2e}")
        print(f"  Max absolute difference: {np.max(abs_diff):.2e}")
        print(f"  Non-zero differences: {np.count_nonzero(abs_diff)}")
        print(f"  Relative difference (max): {np.max(abs_diff) / np.max(csv_matrix) * 100:.2f}%")
        
        # Check if matrices are identical
        if np.allclose(csv_matrix, nc_matrix, rtol=1e-10, atol=1e-15):
            print(f"✅ Matrices are essentially identical!")
        else:
            print(f"⚠️  Matrices differ - this is expected if different data/parameters were used")
            
    else:
        print(f"\n⚠️  Different shapes - cannot compare directly")
        print(f"  CSV: {csv_matrix.shape} vs NetCDF: {nc_matrix.shape}")
        
        # Compare statistics
        print(f"STATISTICAL COMPARISON:")
        print(f"  CSV sparsity: {(1 - np.count_nonzero(csv_matrix)/csv_matrix.size)*100:.2f}%")
        print(f"  NetCDF sparsity: {(1 - np.count_nonzero(nc_matrix)/nc_matrix.size)*100:.2f}%")
        print(f"  CSV max/mean ratio: {np.max(csv_matrix)/np.mean(csv_matrix):.2f}")
        print(f"  NetCDF max/mean ratio: {np.max(nc_matrix)/np.mean(nc_matrix):.2f}")

def plot_comparison(csv_matrix, nc_matrix, save_dir='output/analysis'):
    """Create comparison plots"""
    print(f"\n=== CREATING COMPARISON PLOTS ===")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: CSV matrix
    im1 = axes[0, 0].imshow(np.log10(csv_matrix + 1e-10), cmap='viridis', aspect='equal')
    axes[0, 0].set_title(f'CSV Matrix\nShape: {csv_matrix.shape}')
    axes[0, 0].set_xlabel('Sink Reef')
    axes[0, 0].set_ylabel('Source Reef')
    plt.colorbar(im1, ax=axes[0, 0], label='Log10(Connectivity + 1e-10)')
    
    # Plot 2: NetCDF matrix (subset if too large)
    if nc_matrix.shape[0] > 100:
        # Show subset for visualization
        subset_matrix = nc_matrix[:100, :100]
        title_suffix = f' (subset 100x100)'
    else:
        subset_matrix = nc_matrix
        title_suffix = ''
    
    im2 = axes[0, 1].imshow(np.log10(subset_matrix + 1e-10), cmap='viridis', aspect='equal')
    axes[0, 1].set_title(f'NetCDF Matrix (Moneghetti, sample 0){title_suffix}\nShape: {nc_matrix.shape}')
    axes[0, 1].set_xlabel('Sink Reef')
    axes[0, 1].set_ylabel('Source Reef')
    plt.colorbar(im2, ax=axes[0, 1], label='Log10(Connectivity + 1e-10)')
    
    # Plot 3: Value distributions
    csv_nonzero = csv_matrix[csv_matrix > 0]
    nc_nonzero = nc_matrix[nc_matrix > 0]
    
    axes[1, 0].hist(np.log10(csv_nonzero + 1e-10), bins=30, alpha=0.7, label='CSV', density=True)
    axes[1, 0].hist(np.log10(nc_nonzero + 1e-10), bins=30, alpha=0.7, label='NetCDF', density=True)
    axes[1, 0].set_xlabel('Log10(Connectivity + 1e-10)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Value Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics comparison
    stats_data = [
        ['Metric', 'CSV', 'NetCDF'],
        ['Shape', f'{csv_matrix.shape}', f'{nc_matrix.shape}'],
        ['Non-zero', f'{np.count_nonzero(csv_matrix):,}', f'{np.count_nonzero(nc_matrix):,}'],
        ['Max value', f'{np.max(csv_matrix):.2e}', f'{np.max(nc_matrix):.2e}'],
        ['Mean value', f'{np.mean(csv_matrix):.2e}', f'{np.mean(nc_matrix):.2e}'],
        ['Sparsity', f'{(1-np.count_nonzero(csv_matrix)/csv_matrix.size)*100:.1f}%', 
         f'{(1-np.count_nonzero(nc_matrix)/nc_matrix.size)*100:.1f}%']
    ]
    
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    table = axes[1, 1].table(cellText=stats_data[1:], colLabels=stats_data[0], 
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Statistics Comparison')
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, 'csv_vs_netcdf_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {save_path}")
    
    return fig

def analyze_differences(csv_matrix, nc_matrix):
    """Analyze differences between the matrices"""
    print(f"\n=== DETAILED DIFFERENCE ANALYSIS ===")
    
    if csv_matrix.shape != nc_matrix.shape:
        print("Cannot perform detailed comparison - different shapes")
        return
    
    # Calculate differences
    diff = nc_matrix - csv_matrix
    abs_diff = np.abs(diff)
    
    # Find where differences are largest
    max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    print(f"Largest difference at position {max_diff_idx}:")
    print(f"  CSV value: {csv_matrix[max_diff_idx]:.2e}")
    print(f"  NetCDF value: {nc_matrix[max_diff_idx]:.2e}")
    print(f"  Difference: {diff[max_diff_idx]:.2e}")
    
    # Analyze non-zero differences
    nonzero_diff = abs_diff[abs_diff > 0]
    if len(nonzero_diff) > 0:
        print(f"\nNon-zero differences:")
        print(f"  Count: {len(nonzero_diff)}")
        print(f"  Mean: {np.mean(nonzero_diff):.2e}")
        print(f"  Std: {np.std(nonzero_diff):.2e}")
        print(f"  Max: {np.max(nonzero_diff):.2e}")
        print(f"  Min: {np.min(nonzero_diff):.2e}")
    
    # Check correlation
    correlation = np.corrcoef(csv_matrix.flatten(), nc_matrix.flatten())[0, 1]
    print(f"\nCorrelation coefficient: {correlation:.6f}")
    
    if correlation > 0.99:
        print("✅ Very high correlation - matrices are very similar")
    elif correlation > 0.95:
        print("✅ High correlation - matrices are similar")
    elif correlation > 0.8:
        print("⚠️  Moderate correlation - some differences")
    else:
        print("❌ Low correlation - significant differences")

def main():
    """Main comparison function"""
    print("="*60)
    print("COMPARING CSV vs NETCDF CONNECTIVITY OUTPUT")
    print("="*60)
    
    # File paths
    csv_file = 'output/2015-10-29_Connectivity_max.csv'
    nc_file = 'output/test_parallel_results.nc'
    
    # Load data
    csv_matrix = load_csv_data(csv_file)
    nc_result = load_netcdf_data(nc_file)
    
    if csv_matrix is None or nc_result is None:
        print("❌ Cannot proceed - one or both files not found")
        return
    
    nc_ds, nc_matrix = nc_result
    
    # Compare matrices
    compare_matrices(csv_matrix, nc_matrix)
    
    # Analyze differences
    analyze_differences(csv_matrix, nc_matrix)
    
    # Create comparison plots
    plot_comparison(csv_matrix, nc_matrix)
    
    # Close dataset
    nc_ds.close()
    
    print(f"\n✅ Comparison complete!")
    print(f"Check output/analysis/csv_vs_netcdf_comparison.png for visual comparison")

if __name__ == "__main__":
    main()
