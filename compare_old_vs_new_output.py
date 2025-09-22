#!/usr/bin/env python3
"""
Compare old connectivity_matrices.nc with new test_parallel_results.nc
Focus on Moneghetti treatment from first sample
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_old_file(old_file):
    """Analyze the old connectivity file"""
    print(f"\n=== ANALYZING OLD FILE: {old_file} ===")
    
    if not os.path.exists(old_file):
        print(f"❌ Old file not found: {old_file}")
        return None
    
    old_ds = xr.open_dataset(old_file)
    
    print(f"Dimensions: {dict(old_ds.sizes)}")
    print(f"Variables: {list(old_ds.data_vars)}")
    print(f"Coordinates: {list(old_ds.coords)}")
    
    # Get connectivity data
    if 'time' in old_ds.dims:
        print(f"Time steps: {old_ds.sizes['time']}")
        # Use first time step for comparison
        conn_data = old_ds.connectivity.isel(time=0)
        print(f"Using time step 0 for comparison")
    else:
        conn_data = old_ds.connectivity
    
    matrix = conn_data.values
    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero elements: {np.count_nonzero(matrix)}")
    print(f"Value range: {np.min(matrix):.2e} to {np.max(matrix):.2e}")
    print(f"Mean value: {np.mean(matrix):.2e}")
    
    return old_ds, matrix

def analyze_new_file(new_file):
    """Analyze the new connectivity file (Moneghetti treatment, first sample)"""
    print(f"\n=== ANALYZING NEW FILE: {new_file} ===")
    
    if not os.path.exists(new_file):
        print(f"❌ New file not found: {new_file}")
        return None
    
    new_ds = xr.open_dataset(new_file)
    
    print(f"Dimensions: {dict(new_ds.sizes)}")
    print(f"Variables: {list(new_ds.data_vars)}")
    print(f"Treatments: {list(new_ds.treatment.values)}")
    
    # Get Moneghetti treatment, first sample
    conn_data = new_ds.connectivity.sel(treatment='moneghetti', sample=0)
    matrix = conn_data.values
    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero elements: {np.count_nonzero(matrix)}")
    print(f"Value range: {np.min(matrix):.2e} to {np.max(matrix):.2e}")
    print(f"Mean value: {np.mean(matrix):.2e}")
    
    return new_ds, matrix

def compare_matrices(old_matrix, new_matrix, old_ds, new_ds):
    """Compare the two matrices"""
    print(f"\n=== COMPARING MATRICES ===")
    
    print(f"OLD MATRIX:")
    print(f"  Shape: {old_matrix.shape}")
    print(f"  Non-zero: {np.count_nonzero(old_matrix)}")
    print(f"  Range: {np.min(old_matrix):.2e} to {np.max(old_matrix):.2e}")
    print(f"  Mean: {np.mean(old_matrix):.2e}")
    
    print(f"NEW MATRIX (Moneghetti, sample 0):")
    print(f"  Shape: {new_matrix.shape}")
    print(f"  Non-zero: {np.count_nonzero(new_matrix)}")
    print(f"  Range: {np.min(new_matrix):.2e} to {np.max(new_matrix):.2e}")
    print(f"  Mean: {np.mean(new_matrix):.2e}")
    
    # Check if we can compare directly
    if old_matrix.shape == new_matrix.shape:
        print(f"\n✅ Same shape - can compare directly")
        
        # Calculate differences
        diff = new_matrix - old_matrix
        abs_diff = np.abs(diff)
        
        print(f"DIFFERENCE ANALYSIS:")
        print(f"  Mean difference: {np.mean(diff):.2e}")
        print(f"  Max absolute difference: {np.max(abs_diff):.2e}")
        print(f"  Non-zero differences: {np.count_nonzero(abs_diff)}")
        print(f"  Relative difference (max): {np.max(abs_diff) / np.max(old_matrix) * 100:.2f}%")
        
        # Check if matrices are identical
        if np.allclose(old_matrix, new_matrix, rtol=1e-10, atol=1e-15):
            print(f"✅ Matrices are essentially identical!")
        else:
            print(f"⚠️  Matrices differ - this is expected if different data/parameters were used")
            
    else:
        print(f"\n⚠️  Different shapes - cannot compare directly")
        print(f"  Old: {old_matrix.shape} vs New: {new_matrix.shape}")
        
        # Compare statistics
        print(f"STATISTICAL COMPARISON:")
        print(f"  Old sparsity: {(1 - np.count_nonzero(old_matrix)/old_matrix.size)*100:.2f}%")
        print(f"  New sparsity: {(1 - np.count_nonzero(new_matrix)/new_matrix.size)*100:.2f}%")
        print(f"  Old max/mean ratio: {np.max(old_matrix)/np.mean(old_matrix):.2f}")
        print(f"  New max/mean ratio: {np.max(new_matrix)/np.mean(new_matrix):.2f}")

def plot_comparison(old_matrix, new_matrix, save_dir='output/analysis'):
    """Create comparison plots"""
    print(f"\n=== CREATING COMPARISON PLOTS ===")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Old matrix
    im1 = axes[0, 0].imshow(np.log10(old_matrix + 1e-10), cmap='viridis', aspect='equal')
    axes[0, 0].set_title(f'Old Matrix\nShape: {old_matrix.shape}')
    axes[0, 0].set_xlabel('Target')
    axes[0, 0].set_ylabel('Source')
    plt.colorbar(im1, ax=axes[0, 0], label='Log10(Connectivity + 1e-10)')
    
    # Plot 2: New matrix (subset if too large)
    if new_matrix.shape[0] > 100:
        # Show subset for visualization
        subset_matrix = new_matrix[:100, :100]
        title_suffix = f' (subset 100x100)'
    else:
        subset_matrix = new_matrix
        title_suffix = ''
    
    im2 = axes[0, 1].imshow(np.log10(subset_matrix + 1e-10), cmap='viridis', aspect='equal')
    axes[0, 1].set_title(f'New Matrix (Moneghetti, sample 0){title_suffix}\nShape: {new_matrix.shape}')
    axes[0, 1].set_xlabel('Sink')
    axes[0, 1].set_ylabel('Source')
    plt.colorbar(im2, ax=axes[0, 1], label='Log10(Connectivity + 1e-10)')
    
    # Plot 3: Value distributions
    old_nonzero = old_matrix[old_matrix > 0]
    new_nonzero = new_matrix[new_matrix > 0]
    
    axes[1, 0].hist(np.log10(old_nonzero + 1e-10), bins=30, alpha=0.7, label='Old', density=True)
    axes[1, 0].hist(np.log10(new_nonzero + 1e-10), bins=30, alpha=0.7, label='New', density=True)
    axes[1, 0].set_xlabel('Log10(Connectivity + 1e-10)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Value Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics comparison
    stats_data = [
        ['Metric', 'Old', 'New'],
        ['Shape', f'{old_matrix.shape}', f'{new_matrix.shape}'],
        ['Non-zero', f'{np.count_nonzero(old_matrix):,}', f'{np.count_nonzero(new_matrix):,}'],
        ['Max value', f'{np.max(old_matrix):.2e}', f'{np.max(new_matrix):.2e}'],
        ['Mean value', f'{np.mean(old_matrix):.2e}', f'{np.mean(new_matrix):.2e}'],
        ['Sparsity', f'{(1-np.count_nonzero(old_matrix)/old_matrix.size)*100:.1f}%', 
         f'{(1-np.count_nonzero(new_matrix)/new_matrix.size)*100:.1f}%']
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
    save_path = os.path.join(save_dir, 'old_vs_new_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Comparison plot saved to: {save_path}")
    
    return fig

def main():
    """Main comparison function"""
    print("="*60)
    print("COMPARING OLD vs NEW CONNECTIVITY OUTPUT")
    print("="*60)
    
    # File paths
    old_file = 'output/connectivity_matrices.nc'
    new_file = 'output/test_parallel_results.nc'
    
    # Analyze files
    old_result = analyze_old_file(old_file)
    new_result = analyze_new_file(new_file)
    
    if old_result is None or new_result is None:
        print("❌ Cannot proceed - one or both files not found")
        return
    
    old_ds, old_matrix = old_result
    new_ds, new_matrix = new_result
    
    # Compare matrices
    compare_matrices(old_matrix, new_matrix, old_ds, new_ds)
    
    # Create comparison plots
    plot_comparison(old_matrix, new_matrix)
    
    # Close datasets
    old_ds.close()
    new_ds.close()
    
    print(f"\n✅ Comparison complete!")
    print(f"Check output/analysis/old_vs_new_comparison.png for visual comparison")

if __name__ == "__main__":
    main()
