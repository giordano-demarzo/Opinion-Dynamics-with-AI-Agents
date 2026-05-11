"""
Plot results from opinion dynamics simulation.

This script creates a 3x2 figure comparing different models and opinion pairs
for T=10 simulations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path


def pad_trajectory(history, max_len):
    """
    Pad a trajectory to max_len by repeating the last magnetization value.

    Args:
        history: List of (t, m) tuples
        max_len: Target length for the trajectory

    Returns:
        times: Array of time steps
        magnetizations: Array of magnetization values (padded)
    """
    times = [t for t, m in history]
    mags = [m for t, m in history]

    if len(times) < max_len:
        last_time = times[-1]
        last_mag = mags[-1]

        if len(times) > 1:
            time_increment = times[1] - times[0]
        else:
            time_increment = 1

        for i in range(len(times), max_len):
            times.append(last_time + (i - len(history) + 1) * time_increment)
            mags.append(last_mag)

    return np.array(times), np.array(mags)


def load_data(json_file):
    """Load simulation data from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def process_data(all_results):
    """
    Process simulation data to extract trajectories and final magnetizations.

    Returns:
        m0_values: List of initial magnetization values
        mean_trajectories: Dict mapping m0 to (time_steps, mean_trajectory)
        final_magnetizations: Dict mapping m0 to array of final magnetizations
    """
    m0_values = []
    mean_trajectories = {}
    final_magnetizations = {}

    for m0_str, data in sorted(all_results.items(), key=lambda x: float(x[0])):
        histories = data['histories']
        m0_actual = data['m0_actual']
        m0_values.append(m0_actual)

        # Find max trajectory length
        max_len = max(len(h) for h in histories)

        # Pad all trajectories
        padded_trajectories = []
        time_steps = None
        for history in histories:
            times, mags = pad_trajectory(history, max_len)
            if time_steps is None:
                time_steps = times
            padded_trajectories.append(mags)

        trajectories = np.array(padded_trajectories)
        mean_trajectories[m0_actual] = (time_steps, np.mean(trajectories, axis=0))
        final_magnetizations[m0_actual] = trajectories[:, -1]

    return m0_values, mean_trajectories, final_magnetizations


def plot_magnetization_evolution(ax, m0_values, mean_trajectories, cmap, norm, N=50, show_ylabel=True):
    """Plot collective opinion evolution trajectories."""
    for m0 in m0_values:
        color = cmap(norm(m0))
        time_steps, mean_traj = mean_trajectories[m0]
        # Normalize time by system size N
        normalized_time = time_steps / N
        ax.plot(normalized_time, mean_traj, color=color, linewidth=2)

    ax.set_xlabel('Time t', fontsize=13)
    if show_ylabel:
        ax.set_ylabel('Collective opinion m(t)', fontsize=13)
    ax.tick_params(axis='both', labelsize=11)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, 10)


def plot_final_distribution(ax, m0_values, final_magnetizations, cmap, norm, show_ylabel=True):
    """Plot final collective opinion distributions (x = m0, y = m_f)."""
    x_positions = np.arange(len(m0_values))
    x_spacing = 0.4

    # Find the x position for m0=0
    m0_zero_x_pos = None
    for idx, m0 in enumerate(m0_values):
        if abs(m0) < 0.01:
            m0_zero_x_pos = x_positions[idx]

    for idx, m0 in enumerate(m0_values):
        final_mags = final_magnetizations[m0]
        color = cmap(norm(m0))
        x_pos = x_positions[idx]

        # Create histogram bins along the y (final opinion) axis
        bins = np.linspace(-1.05, 1.05, 41)
        counts, bin_edges = np.histogram(final_mags, bins=bins)

        # Normalize counts
        if counts.max() > 0:
            counts_scaled = counts / counts.max() * x_spacing
        else:
            counts_scaled = counts

        # Plot histogram bars extending left and right from x_pos
        for i, count in enumerate(counts_scaled):
            if count > 0:
                rect_y = [bin_edges[i], bin_edges[i], bin_edges[i+1], bin_edges[i+1]]
                rect_x_r = [x_pos, x_pos + count, x_pos + count, x_pos]
                rect_x_l = [x_pos, x_pos - count, x_pos - count, x_pos]
                ax.fill(rect_x_r, rect_y, color=color, alpha=0.6, edgecolor=color, linewidth=0.5)
                ax.fill(rect_x_l, rect_y, color=color, alpha=0.6, edgecolor=color, linewidth=0.5)

        # Add scatter points with horizontal jitter
        jitter = np.random.normal(0, 0.03, size=len(final_mags))
        ax.scatter(x_pos + jitter, final_mags, color=color, alpha=0.7, s=15,
                   edgecolors='black', linewidths=0.2, zorder=3)

    # Configure axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{m0:+.1f}' for m0 in m0_values], fontsize=9, rotation=45, ha='right')
    ax.set_xlabel(r'Initial collective opinion $m_0$', fontsize=13)
    if show_ylabel:
        ax.set_ylabel(r'Final collective opinion $m_f$', fontsize=13)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)

    # Add vertical line at m0=0 position
    if m0_zero_x_pos is not None:
        ax.axvline(x=m0_zero_x_pos, color='k', linestyle='--', alpha=0.3, linewidth=1)

    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(-0.5, len(x_positions) - 0.5)


def main():
    # Define the data files for T=10
    base_path = Path(__file__).parent.parent / 'data'

    data_configs = [
        {
            'file': base_path / 'gemma-3-27b-it/gender self-identification/simulation_data_N50_T10.json',
            'model': 'Gemma 3 27B',
            'opinion': 'Gender self-identification\nvs Biological sex classification'
        },
        {
            'file': base_path / 'gemma-3-27b-it/renewable energy/simulation_data_N50_T10.json',
            'model': 'Gemma 3 27B',
            'opinion': 'Renewable energy\nvs Fossil fuels'
        },
        {
            'file': base_path / 'Llama-3.1-8B-Instruct/gender self-identification/simulation_data_N50_T10.json',
            'model': 'Llama 3.1 8B',
            'opinion': 'Gender self-identification\nvs Biological sex classification'
        }
    ]

    # Create figure with 3 columns and 2 rows
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Set up colormap centered at m=0
    cmap = plt.cm.coolwarm
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    print("Loading and processing data...")

    for col_idx, config in enumerate(data_configs):
        print(f"  Processing: {config['model']} - {config['opinion'].replace(chr(10), ' ')}")

        # Load data
        all_results = load_data(config['file'])

        # Process data
        m0_values, mean_trajectories, final_magnetizations = process_data(all_results)

        # Only show y-labels in the first column
        show_ylabel = (col_idx == 0)

        # Plot magnetization evolution (row 0)
        ax_evolution = axes[0, col_idx]
        plot_magnetization_evolution(ax_evolution, m0_values, mean_trajectories, cmap, norm, show_ylabel=show_ylabel)

        # Plot final distribution (row 1)
        ax_distribution = axes[1, col_idx]
        plot_final_distribution(ax_distribution, m0_values, final_magnetizations, cmap, norm, show_ylabel=show_ylabel)

        # Add column title - model name bold, opinion name not bold
        axes[0, col_idx].set_title(config['model'], fontsize=12, fontweight='bold', pad=35)
        # Add opinion as text below the title (not bold)
        axes[0, col_idx].text(0.5, 1.04, config['opinion'], transform=axes[0, col_idx].transAxes,
                              fontsize=11, ha='center', va='bottom')

    # Add row labels on the left
    #axes[0, 0].annotate('Collective opinion\nEvolution', xy=(-0.35, 0.5), xycoords='axes fraction', fontsize=14, fontweight='bold', ha='center', va='center', rotation=90)
    #axes[1, 0].annotate('Final\nDistributions', xy=(-0.35, 0.5), xycoords='axes fraction', fontsize=14, fontweight='bold', ha='center', va='center', rotation=90)

    # Adjust layout first, then add colorbar
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.88, wspace=0.3)

    # Add common colorbar
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Initial collective opinion (m₀)', fontsize=13)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    cbar.ax.tick_params(labelsize=11)

    # Save plots
    output_dir = base_path
    plot_path_png = output_dir / "comparison_plot_T10.png"
    plot_path_pdf = output_dir / "comparison_plot_T10.pdf"

    plt.savefig(plot_path_png, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path_png}")

    plt.savefig(plot_path_pdf, bbox_inches='tight')
    print(f"Plot (PDF) saved to: {plot_path_pdf}")

    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
