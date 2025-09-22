#!/usr/bin/env python3
"""
Analyze connectivity matrices from the test output
Plot full 3806x3806 matrices for both treatments (first sample only)
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
import os

def plot_connectivity_matrix(ds, treatment, sample_idx=0, max_reefs=None, save_path=None):
    """
    Plot the full connectivity matrix for a specific treatment and sample
    """
    print(f"\n=== Plotting {treatment} treatment (sample {sample_idx}) ===")
    
    # Get connectivity data for this treatment and sample
    conn_data = ds.connectivity.sel(treatment=treatment, sample=sample_idx)
    
    # Convert to numpy array
    matrix = conn_data.values
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Non-zero values: {np.count_nonzero(matrix)}")
    print(f"Max value: {np.max(matrix):.8f}")
    print(f"Min value: {np.min(matrix):.8f}")
    print(f"Mean value: {np.mean(matrix):.8f}")
    
    # If max_reefs is specified, subset the matrix
    if max_reefs is not None:
        matrix = matrix[:max_reefs, :max_reefs]
        print(f"Subset to {max_reefs}x{max_reefs} for visualization")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Use a logarithmic scale for better visualization
    # Add small value to avoid log(0)
    log_matrix = np.log10(matrix + 1e-10)
    
    # Create the heatmap
    im = ax.imshow(log_matrix, cmap='viridis', aspect='equal', 
                   origin='lower', interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Log10(Connectivity + 1e-10)', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_xlabel('Sink Reef ID', fontsize=12)
    ax.set_ylabel('Source Reef ID', fontsize=12)
    ax.set_title(f'Connectivity Matrix - {treatment.title()} Treatment (Sample {sample_idx})\n'
                f'Matrix size: {matrix.shape[0]}×{matrix.shape[1]}', 
                fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add statistics text box
    stats_text = (f'Non-zero: {np.count_nonzero(matrix):,}\n'
                 f'Max: {np.max(matrix):.2e}\n'
                 f'Mean: {np.mean(matrix):.2e}\n'
                 f'Min: {np.min(matrix):.2e}')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.8), fontsize=10)
    
    # Highlight diagonal (self-connectivity)
    if matrix.shape[0] == matrix.shape[1]:
        ax.add_patch(Rectangle((0, 0), matrix.shape[0]-1, matrix.shape[1]-1, 
                              fill=False, edgecolor='red', linewidth=2, linestyle='--'))
        ax.text(0.5, 0.95, 'Diagonal = Self-connectivity', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3), 
                fontsize=10, color='white')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    return fig, ax

def plot_connectivity_comparison(ds, sample_idx=0, max_reefs=None, save_path=None):
    """
    Plot both treatments side by side for comparison
    """
    print(f"\n=== Plotting comparison (sample {sample_idx}) ===")
    
    treatments = ['moneghetti', 'connolly']
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    for i, treatment in enumerate(treatments):
        # Get connectivity data
        conn_data = ds.connectivity.sel(treatment=treatment, sample=sample_idx)
        matrix = conn_data.values
        
        # Subset if needed
        if max_reefs is not None:
            matrix = matrix[:max_reefs, :max_reefs]
        
        # Log scale
        log_matrix = np.log10(matrix + 1e-10)
        
        # Plot
        im = axes[i].imshow(log_matrix, cmap='viridis', aspect='equal', 
                           origin='lower', interpolation='nearest')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
        cbar.set_label('Log10(Connectivity + 1e-10)', rotation=270, labelpad=20)
        
        # Labels
        axes[i].set_xlabel('Sink Reef ID', fontsize=12)
        axes[i].set_ylabel('Source Reef ID', fontsize=12)
        axes[i].set_title(f'{treatment.title()} Treatment\n'
                         f'Non-zero: {np.count_nonzero(matrix):,}\n'
                         f'Max: {np.max(matrix):.2e}', 
                         fontsize=14, fontweight='bold')
        
        # Grid
        axes[i].grid(True, alpha=0.3, linewidth=0.5)
        
        # Highlight diagonal
        if matrix.shape[0] == matrix.shape[1]:
            axes[i].add_patch(Rectangle((0, 0), matrix.shape[0]-1, matrix.shape[1]-1, 
                                      fill=False, edgecolor='red', linewidth=2, linestyle='--'))
    
    plt.suptitle(f'Connectivity Matrix Comparison (Sample {sample_idx})\n'
                f'Matrix size: {matrix.shape[0]}×{matrix.shape[1]}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Comparison plot saved to: {save_path}")
    
    return fig, axes

def analyze_connectivity_patterns(ds, sample_idx=0):
    """
    Analyze connectivity patterns and statistics
    """
    print(f"\n=== Connectivity Pattern Analysis (Sample {sample_idx}) ===")
    
    treatments = ['moneghetti', 'connolly']
    
    for treatment in treatments:
        print(f"\n--- {treatment.upper()} TREATMENT ---")
        
        # Get data
        conn_data = ds.connectivity.sel(treatment=treatment, sample=sample_idx)
        matrix = conn_data.values
        
        # Basic statistics
        print(f"Matrix shape: {matrix.shape}")
        print(f"Total elements: {matrix.size:,}")
        print(f"Non-zero elements: {np.count_nonzero(matrix):,}")
        print(f"Sparsity: {(1 - np.count_nonzero(matrix)/matrix.size)*100:.2f}%")
        
        # Value statistics
        non_zero_values = matrix[matrix > 0]
        if len(non_zero_values) > 0:
            print(f"Non-zero value range: {np.min(non_zero_values):.2e} to {np.max(non_zero_values):.2e}")
            print(f"Non-zero value mean: {np.mean(non_zero_values):.2e}")
            print(f"Non-zero value std: {np.std(non_zero_values):.2e}")
        
        # Self-connectivity analysis
        diagonal = np.diag(matrix)
        self_conn_nonzero = diagonal[diagonal > 0]
        print(f"Self-connectivity: {len(self_conn_nonzero)}/{len(diagonal)} reefs have self-connectivity")
        if len(self_conn_nonzero) > 0:
            print(f"Self-connectivity range: {np.min(self_conn_nonzero):.2e} to {np.max(self_conn_nonzero):.2e}")
            print(f"Self-connectivity mean: {np.mean(self_conn_nonzero):.2e}")
        
        # Outgoing connectivity (row sums)
        outgoing = np.sum(matrix, axis=1)
        outgoing_nonzero = outgoing[outgoing > 0]
        print(f"Outgoing connectivity: {len(outgoing_nonzero)}/{len(outgoing)} reefs have outgoing connections")
        if len(outgoing_nonzero) > 0:
            print(f"Outgoing range: {np.min(outgoing_nonzero):.2e} to {np.max(outgoing_nonzero):.2e}")
            print(f"Outgoing mean: {np.mean(outgoing_nonzero):.2e}")
        
        # Incoming connectivity (column sums)
        incoming = np.sum(matrix, axis=0)
        incoming_nonzero = incoming[incoming > 0]
        print(f"Incoming connectivity: {len(incoming_nonzero)}/{len(incoming)} reefs have incoming connections")
        if len(incoming_nonzero) > 0:
            print(f"Incoming range: {np.min(incoming_nonzero):.2e} to {np.max(incoming_nonzero):.2e}")
            print(f"Incoming mean: {np.mean(incoming_nonzero):.2e}")

def main():
    """Main analysis function"""
    print("="*60)
    print("CONNECTIVITY MATRIX ANALYSIS")
    print("="*60)
    
    # Load the dataset
    nc_file = 'output/test_parallel_results.nc'
    if not os.path.exists(nc_file):
        print(f"❌ Error: {nc_file} not found!")
        return
    
    print(f"Loading data from: {nc_file}")
    ds = xr.open_dataset(nc_file)
    
    print(f"Dataset dimensions: {dict(ds.sizes)}")
    print(f"Treatments: {list(ds.treatment.values)}")
    print(f"Number of samples: {ds.sizes['sample']}")
    
    # Analyze patterns
    analyze_connectivity_patterns(ds, sample_idx=0)
    
    # Create output directory
    os.makedirs('output/analysis', exist_ok=True)
    
    # Plot individual matrices (full size)
    print(f"\n=== Creating full matrix plots ===")
    plot_connectivity_matrix(ds, 'moneghetti', sample_idx=0, 
                           save_path='output/analysis/connectivity_matrix_moneghetti_full.png')
    plot_connectivity_matrix(ds, 'connolly', sample_idx=0, 
                           save_path='output/analysis/connectivity_matrix_connolly_full.png')
    
    # Plot comparison
    plot_connectivity_comparison(ds, sample_idx=0, 
                               save_path='output/analysis/connectivity_comparison_full.png')
    
    # Plot subset for better visualization (first 1000x1000)
    print(f"\n=== Creating subset plots (1000x1000) ===")
    plot_connectivity_matrix(ds, 'moneghetti', sample_idx=0, max_reefs=1000,
                           save_path='output/analysis/connectivity_matrix_moneghetti_subset.png')
    plot_connectivity_matrix(ds, 'connolly', sample_idx=0, max_reefs=1000,
                           save_path='output/analysis/connectivity_matrix_connolly_subset.png')
    plot_connectivity_comparison(ds, sample_idx=0, max_reefs=1000,
                               save_path='output/analysis/connectivity_comparison_subset.png')
    
    # Close dataset
    ds.close()
    
    print(f"\n✅ Analysis complete! Check output/analysis/ directory for plots.")
    print(f"Files created:")
    print(f"  - connectivity_matrix_moneghetti_full.png")
    print(f"  - connectivity_matrix_connolly_full.png") 
    print(f"  - connectivity_comparison_full.png")
    print(f"  - connectivity_matrix_moneghetti_subset.png")
    print(f"  - connectivity_matrix_connolly_subset.png")
    print(f"  - connectivity_comparison_subset.png")

if __name__ == "__main__":
    main()
