#!/usr/bin/env python3
"""
Plot Connectivity Results
========================

Plot connectivity results for the specific reefs that were processed.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    sns.set_palette("husl")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Note: seaborn not available, using default matplotlib style")

# Set up plotting style
plt.style.use('default')


def plot_connectivity_summary(nc_file, reef_ids=[0, 1, 2, 3]):
    """
    Plot connectivity results for specific reefs.
    
    Parameters
    ----------
    nc_file : str
        Path to NetCDF file with connectivity results.
    reef_ids : list
        List of reef IDs to plot.
    """
    print("="*60)
    print("PLOTTING CONNECTIVITY RESULTS")
    print("="*60)
    
    # Load the data
    ds = xr.open_dataset(nc_file)
    print(f"Loaded data from: {nc_file}")
    print(f"Dimensions: {dict(ds.dims)}")
    
    # Extract data for specific reefs
    reef_subset = ds.sel(source=reef_ids, sink=reef_ids)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Connectivity Results for Reefs {reef_ids}', fontsize=16, fontweight='bold')
    
    # 1. Mean connectivity matrix
    print("1. Plotting mean connectivity matrix...")
    connectivity_mean = reef_subset['connectivity'].mean(dim='sample')
    im1 = axes[0, 0].imshow(connectivity_mean.values, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Mean Connectivity Matrix')
    axes[0, 0].set_xlabel('Sink Reef')
    axes[0, 0].set_ylabel('Source Reef')
    plt.colorbar(im1, ax=axes[0, 0], label='Connectivity')
    
    # Add reef labels
    axes[0, 0].set_xticks(range(len(reef_ids)))
    axes[0, 0].set_yticks(range(len(reef_ids)))
    axes[0, 0].set_xticklabels([f'R{i}' for i in reef_ids])
    axes[0, 0].set_yticklabels([f'R{i}' for i in reef_ids])
    
    # 2. Connectivity variance
    print("2. Plotting connectivity variance...")
    connectivity_var = reef_subset['connectivity'].var(dim='sample')
    im2 = axes[0, 1].imshow(connectivity_var.values, cmap='plasma', aspect='auto')
    axes[0, 1].set_title('Connectivity Variance')
    axes[0, 1].set_xlabel('Sink Reef')
    axes[0, 1].set_ylabel('Source Reef')
    plt.colorbar(im2, ax=axes[0, 1], label='Variance')
    
    # Add reef labels
    axes[0, 1].set_xticks(range(len(reef_ids)))
    axes[0, 1].set_yticks(range(len(reef_ids)))
    axes[0, 1].set_xticklabels([f'R{i}' for i in reef_ids])
    axes[0, 1].set_yticklabels([f'R{i}' for i in reef_ids])
    
    # 3. Distance matrix
    print("3. Plotting distance matrix...")
    distance_matrix = reef_subset['distance'].values
    im3 = axes[0, 2].imshow(distance_matrix, cmap='Blues', aspect='auto')
    axes[0, 2].set_title('Distance Matrix (km)')
    axes[0, 2].set_xlabel('Sink Reef')
    axes[0, 2].set_ylabel('Source Reef')
    plt.colorbar(im3, ax=axes[0, 2], label='Distance (km)')
    
    # Add reef labels
    axes[0, 2].set_xticks(range(len(reef_ids)))
    axes[0, 2].set_yticks(range(len(reef_ids)))
    axes[0, 2].set_xticklabels([f'R{i}' for i in reef_ids])
    axes[0, 2].set_yticklabels([f'R{i}' for i in reef_ids])
    
    # 4. Bootstrap connectivity distributions
    print("4. Plotting bootstrap distributions...")
    for i, reef_id in enumerate(reef_ids):
        # Get self-connectivity bootstrap samples
        self_connectivity = reef_subset['connectivity'].sel(source=reef_id, sink=reef_id).values
        axes[1, 0].hist(self_connectivity, alpha=0.7, label=f'Reef {reef_id}', bins=20)
    
    axes[1, 0].set_title('Self-Connectivity Distributions')
    axes[1, 0].set_xlabel('Connectivity')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Connectivity vs Distance
    print("5. Plotting connectivity vs distance...")
    # Get all pairwise connections (excluding self-connections)
    source_reefs = []
    sink_reefs = []
    distances = []
    connectivities = []
    
    for i, source_id in enumerate(reef_ids):
        for j, sink_id in enumerate(reef_ids):
            if i != j:  # Exclude self-connections
                dist = reef_subset['distance'].sel(source=source_id, sink=sink_id).values
                conn = reef_subset['connectivity'].sel(source=source_id, sink=sink_id).mean(dim='sample').values
                distances.append(dist)
                connectivities.append(conn)
                source_reefs.append(source_id)
                sink_reefs.append(sink_id)
    
    axes[1, 1].scatter(distances, connectivities, alpha=0.7, s=100)
    axes[1, 1].set_title('Connectivity vs Distance')
    axes[1, 1].set_xlabel('Distance (km)')
    axes[1, 1].set_ylabel('Mean Connectivity')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, (dist, conn, src, snk) in enumerate(zip(distances, connectivities, source_reefs, sink_reefs)):
        axes[1, 1].annotate(f'{src}→{snk}', (dist, conn), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
    
    # 6. Summary statistics
    print("6. Creating summary statistics...")
    axes[1, 2].axis('off')
    
    # Calculate summary statistics
    summary_text = "SUMMARY STATISTICS\n\n"
    
    for reef_id in reef_ids:
        # Self-connectivity
        self_conn = reef_subset['connectivity'].sel(source=reef_id, sink=reef_id)
        self_mean = self_conn.mean(dim='sample').values
        self_std = self_conn.std(dim='sample').values
        
        # Outgoing connectivity
        outgoing = reef_subset['connectivity'].sel(source=reef_id)
        outgoing_mean = outgoing.mean(dim=['sink', 'sample']).values
        
        # Incoming connectivity
        incoming = reef_subset['connectivity'].sel(sink=reef_id)
        incoming_mean = incoming.mean(dim=['source', 'sample']).values
        
        summary_text += f"Reef {reef_id}:\n"
        summary_text += f"  Self-connectivity: {self_mean:.6f} ± {self_std:.6f}\n"
        summary_text += f"  Outgoing mean: {outgoing_mean:.6f}\n"
        summary_text += f"  Incoming mean: {incoming_mean:.6f}\n\n"
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "output/connectivity_plots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    ds.close()
    
    return fig


def plot_individual_reef_analysis(nc_file, reef_ids=[0, 1, 2, 3]):
    """
    Create detailed individual reef analysis plots.
    """
    print("\n" + "="*60)
    print("INDIVIDUAL REEF ANALYSIS")
    print("="*60)
    
    ds = xr.open_dataset(nc_file)
    reef_subset = ds.sel(source=reef_ids, sink=reef_ids)
    
    # Create individual plots for each reef
    for reef_id in reef_ids:
        print(f"Creating detailed plot for Reef {reef_id}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Detailed Analysis: Reef {reef_id}', fontsize=14, fontweight='bold')
        
        # 1. Outgoing connectivity
        outgoing = reef_subset['connectivity'].sel(source=reef_id).mean(dim='sample')
        im1 = axes[0, 0].imshow(outgoing.values.reshape(1, -1), cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'Outgoing Connectivity from Reef {reef_id}')
        axes[0, 0].set_xlabel('Sink Reef')
        axes[0, 0].set_ylabel('Source Reef')
        plt.colorbar(im1, ax=axes[0, 0], label='Connectivity')
        
        # Add reef labels
        axes[0, 0].set_xticks(range(len(reef_ids)))
        axes[0, 0].set_xticklabels([f'R{i}' for i in reef_ids])
        axes[0, 0].set_yticks([0])
        axes[0, 0].set_yticklabels([f'R{reef_id}'])
        
        # 2. Incoming connectivity
        incoming = reef_subset['connectivity'].sel(sink=reef_id).mean(dim='sample')
        im2 = axes[0, 1].imshow(incoming.values.reshape(-1, 1), cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'Incoming Connectivity to Reef {reef_id}')
        axes[0, 1].set_xlabel('Sink Reef')
        axes[0, 1].set_ylabel('Source Reef')
        plt.colorbar(im2, ax=axes[0, 1], label='Connectivity')
        
        # Add reef labels
        axes[0, 1].set_xticks([0])
        axes[0, 1].set_xticklabels([f'R{reef_id}'])
        axes[0, 1].set_yticks(range(len(reef_ids)))
        axes[0, 1].set_yticklabels([f'R{i}' for i in reef_ids])
        
        # 3. Bootstrap distribution for self-connectivity
        self_conn = reef_subset['connectivity'].sel(source=reef_id, sink=reef_id).values
        axes[1, 0].hist(self_conn, bins=20, alpha=0.7, color='green')
        axes[1, 0].set_title(f'Self-Connectivity Distribution (Reef {reef_id})')
        axes[1, 0].set_xlabel('Connectivity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(self_conn)
        std_val = np.std(self_conn)
        axes[1, 0].axvline(mean_val, color='red', linestyle='--', 
                           label=f'Mean: {mean_val:.6f}')
        axes[1, 0].legend()
        
        # 4. Connectivity vs Distance for this reef
        distances = []
        connectivities = []
        
        for sink_id in reef_ids:
            if reef_id != sink_id:
                dist = reef_subset['distance'].sel(source=reef_id, sink=sink_id).values
                conn = reef_subset['connectivity'].sel(source=reef_id, sink=sink_id).mean(dim='sample').values
                distances.append(dist)
                connectivities.append(conn)
        
        axes[1, 1].scatter(distances, connectivities, s=100, alpha=0.7)
        axes[1, 1].set_title(f'Outgoing Connectivity vs Distance (Reef {reef_id})')
        axes[1, 1].set_xlabel('Distance (km)')
        axes[1, 1].set_ylabel('Mean Connectivity')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add labels
        for i, (dist, conn) in enumerate(zip(distances, connectivities)):
            sink_id = [rid for rid in reef_ids if rid != reef_id][i]
            axes[1, 1].annotate(f'→{sink_id}', (dist, conn), xytext=(5, 5), 
                               textcoords='offset points', fontsize=10)
        
        plt.tight_layout()
        
        # Save individual plot
        output_file = f"output/reef_{reef_id}_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Individual plot saved to: {output_file}")
        
        plt.show()
    
    ds.close()


def main():
    """Main function to create all plots."""
    nc_file = "output/connectivity_results_example.nc"
    
    if not Path(nc_file).exists():
        print(f"❌ NetCDF file not found: {nc_file}")
        print("Please run the connectivity analysis first.")
        return False
    
    # Create output directory for plots
    Path("output").mkdir(exist_ok=True)
    
    # Create summary plots
    plot_connectivity_summary(nc_file)
    
    # Create individual reef analysis
    plot_individual_reef_analysis(nc_file)
    
    print("\n" + "="*60)
    print("PLOTTING COMPLETE")
    print("="*60)
    print("✅ All plots have been created and saved to the output/ directory")
    
    return True


if __name__ == "__main__":
    main() 