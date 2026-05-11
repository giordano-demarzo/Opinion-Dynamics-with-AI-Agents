"""
Three-panel figure showing phase diagrams:
1. Left: Phase diagram of Gemma 3 27B (all data points with some labels)
2. Center: Phase diagram with all models (median points per model with labels)
3. Right: Phase diagram with flipping probability (scatter plot like metastability_all_models.pdf)

Uses consistent style with plot1.py and plot2.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.lines import Line2D
from pathlib import Path

# Set plotting style (matching other plots)
rcParams.update({
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
})

BASE_PATH = str(Path(__file__).parent.parent / 'data')


def find_spinodal_infinite(beta):
    """
    Find spinodal boundary for infinite system.
    Returns |h_spinodal| for given beta.
    """
    if beta <= 1:
        return None
    m_spinodal = np.sqrt(1 - 1 / beta)
    h_spinodal = -m_spinodal + np.arctanh(m_spinodal) / beta
    return np.abs(h_spinodal)


def get_spinodal_line(beta_max=10):
    """Generate spinodal line data."""
    beta_range = np.linspace(1.01, beta_max, 1000)
    h_spinodal = np.array([find_spinodal_infinite(b) if find_spinodal_infinite(b) is not None else np.nan
                          for b in beta_range])
    valid_mask = ~np.isnan(h_spinodal)
    return beta_range[valid_mask], h_spinodal[valid_mask]


def get_model_family(name):
    """Determine model family for coloring."""
    name_lower = name.lower()
    if 'llama' in name_lower:
        return 'Llama'
    elif 'gemma' in name_lower or 'gemma-3' in name_lower:
        return 'Gemma'
    elif 'gemini' in name_lower:
        return 'Gemini'
    elif 'qwen3' in name_lower or 'qwen-3' in name_lower:
        return 'Qwen-3'
    elif 'qwen2.5' in name_lower or 'qwen-2.5' in name_lower:
        return 'Qwen-2.5'
    elif 'mistral' in name_lower:
        return 'Mistral'
    elif 'gpt' in name_lower:
        return 'GPT'
    elif 'grok' in name_lower:
        return 'Grok'
    else:
        return 'Other'


def get_short_model_name(name):
    """Get shortened model name for labels."""
    name_lower = name.lower()
    if 'llama-3.1-8b' in name_lower:
        return 'Llama-3.1-8B'
    elif 'gemma-3-27b' in name_lower:
        return 'Gemma-3-27B'
    elif 'gemma-3-12b' in name_lower:
        return 'Gemma-3-12B'
    elif 'gemini-2.5-flash' in name_lower:
        return 'Gemini-2.5-Flash'
    elif 'qwen3-32b' in name_lower:
        return 'Qwen3-32B'
    elif 'qwen3-14b' in name_lower:
        return 'Qwen3-14B'
    elif 'qwen2.5-32b' in name_lower:
        return 'Qwen2.5-32B'
    elif 'qwen2.5-14b' in name_lower:
        return 'Qwen2.5-14B'
    elif 'gpt-5-mini' in name_lower:
        return 'GPT-5-Mini'
    else:
        return name


# Color and marker settings for individual models (matching plot2.py)
MODEL_COLORS = {
    'Gemma-3-27B': '#1f77b4',
    'Gemma-3-12B': '#aec7e8',
    'Llama-3.1-8B': '#ff7f0e',
    'Qwen2.5-14B': '#2ca02c',
    'Qwen2.5-32B': '#98df8a',
    'Qwen3-14B': '#d62728',
    'Qwen3-32B': '#ff9896',
    'Gemini-2.5-Flash': '#9467bd',
    'GPT-5-Mini': '#8c564b',
}

MODEL_MARKERS = {
    'Gemma-3-27B': 'o',
    'Gemma-3-12B': 's',
    'Llama-3.1-8B': '^',
    'Qwen2.5-14B': 'D',
    'Qwen2.5-32B': 'v',
    'Qwen3-14B': 'p',
    'Qwen3-32B': 'h',
    'Gemini-2.5-Flash': '*',
    'GPT-5-Mini': 'X',
}


def filter_by_opinion_pairs(df, opinion_pairs_df, opinion_col_A='Opinion_A', opinion_col_B='Opinion_B'):
    """
    Filter dataframe to only include opinion pairs from opinion_pairs.csv.
    Similar to plot2.py filtering logic.
    """
    # Create set of valid opinion pairs (normalized to lowercase)
    valid_pairs = set()
    for _, row in opinion_pairs_df.iterrows():
        op_a = str(row['Opinion_A']).lower().strip()
        op_b = str(row['Opinion_B']).lower().strip()
        valid_pairs.add((op_a, op_b))
        valid_pairs.add((op_b, op_a))  # Both orderings

    # Filter the dataframe
    mask = []
    for _, row in df.iterrows():
        op_a = str(row[opinion_col_A]).lower().strip()
        op_b = str(row[opinion_col_B]).lower().strip()
        mask.append((op_a, op_b) in valid_pairs)

    return df[mask].copy()


def main():
    # Load opinion pairs for filtering
    opinion_pairs = pd.read_csv(f'{BASE_PATH}/opinion_pairs.csv')
    print(f"Opinion pairs in CSV: {len(opinion_pairs)}")

    # Load data from correct sources
    print("Loading data...")

    # For Gemma panel: use all_fits_combined.csv (has correct column names)
    all_fits = pd.read_csv(f'{BASE_PATH}/analysis_results_v2/all_fits_combined.csv')
    all_fits['abs_h'] = np.abs(all_fits['h'])

    # For metastability panel: use metastability_results.csv
    metastability = pd.read_csv(f'{BASE_PATH}/metastability_results.csv')
    metastability['abs_h'] = np.abs(metastability['h'])

    # Get spinodal line
    beta_spinodal, h_spinodal = get_spinodal_line(beta_max=15)

    # Create figure: 3 equal panels + 1 narrow colorbar column
    fig = plt.figure(figsize=(18, 5.5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.35)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    cbar_ax = fig.add_subplot(gs[3])

    # Set style for all plots (matching plot1.py and plot2.py)
    for ax in axes:
        ax.grid(True, alpha=0.3)

    # ===== LEFT PANEL: Gemma 3 27B phase diagram =====
    ax1 = axes[0]

    # Filter for Gemma-27B and filter by opinion pairs
    gemma_data = all_fits[all_fits['Model'].str.contains('gemma-3-27b', case=False)].copy()
    gemma_data = filter_by_opinion_pairs(gemma_data, opinion_pairs)
    print(f"Gemma-27B data points (filtered): {len(gemma_data)}")

    # Plot spinodal region (metastable)
    ax1.fill_between(beta_spinodal, 0, h_spinodal, color='gray',
                     alpha=0.15, zorder=0, label='Metastable region')

    # Plot spinodal line
    ax1.plot(beta_spinodal, h_spinodal, 'k--', linewidth=2.5,
             label='Spinodal boundary', zorder=10)

    # Plot Gemma data points
    ax1.scatter(gemma_data['beta'], gemma_data['abs_h'],
                s=60, alpha=0.6, color='#1f77b4',
                edgecolors='white', linewidths=0.5, zorder=5)

    # Add labels for specific opinion pairs (same as plot1.py)
    target_opinions = [
        ('renewable energy', 'fossil fuels'),
        ('gender self-identification', 'biological sex classification'),
    ]

    # Find and label the specific opinion pairs
    for target_a, target_b in target_opinions:
        # Search for matching opinion pair
        mask = (
            (gemma_data['Opinion_A'].str.lower().str.contains(target_a.lower())) |
            (gemma_data['Opinion_B'].str.lower().str.contains(target_a.lower()))
        )
        matching = gemma_data[mask]

        if len(matching) > 0:
            point = matching.iloc[0]
            beta = point['beta']
            h = point['abs_h']
            op_a = point['Opinion_A']
            op_b = point['Opinion_B']

            label_text = f"{op_a}\nvs {op_b}"

            # Position labels based on content
            if 'renewable' in op_a.lower() or 'renewable' in op_b.lower():
                xytext = (-150, 15)
                ha = 'left'
            else:  # gender self-identification
                xytext = (-25, -35)
                ha = 'left'

            ax1.annotate(label_text,
                        (beta, h),
                        xytext=xytext, textcoords='offset points',
                        fontsize=12,
                        ha=ha, va='center',
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white', alpha=0.8,
                                edgecolor='#1f77b4', linewidth=1),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                      color='black', lw=1),
                        zorder=15)

    ax1.set_xlabel(r'Majority force $\beta$', fontsize=16)
    ax1.set_ylabel(r'Bias magnitude $|h|$', fontsize=16)
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 1.5)
    ax1.legend(loc='upper right', fontsize=11)

    # ===== CENTER PANEL: All models median phase diagram =====
    ax2 = axes[1]

    # Filter all_fits by opinion pairs and compute medians
    all_fits_filtered = filter_by_opinion_pairs(all_fits, opinion_pairs)
    print(f"All fits (filtered): {len(all_fits_filtered)}")

    model_medians = []
    for model_name in all_fits_filtered['Model'].unique():
        model_df = all_fits_filtered[all_fits_filtered['Model'] == model_name].copy()
        if len(model_df) > 0:
            median_beta = model_df['beta'].median()
            median_h = model_df['abs_h'].median()

            # Calculate 25th and 75th percentiles for error bars
            q25_beta = model_df['beta'].quantile(0.25)
            q75_beta = model_df['beta'].quantile(0.75)
            q25_h = model_df['abs_h'].quantile(0.25)
            q75_h = model_df['abs_h'].quantile(0.75)

            family = get_model_family(model_name)
            short_name = get_short_model_name(model_name)
            model_medians.append({
                'model': model_name,
                'short_name': short_name,
                'beta': median_beta,
                'h': median_h,
                'beta_err_lower': median_beta - q25_beta,
                'beta_err_upper': q75_beta - median_beta,
                'h_err_lower': median_h - q25_h,
                'h_err_upper': q75_h - median_h,
                'family': family
            })

    median_df = pd.DataFrame(model_medians)
    print(f"Models with median values: {len(median_df)}")
    for _, row in median_df.iterrows():
        print(f"  {row['model']}: beta={row['beta']:.2f}, |h|={row['h']:.3f}")

    # Plot spinodal region (no label - already in left panel)
    ax2.fill_between(beta_spinodal, 0, h_spinodal, color='gray',
                     alpha=0.15, zorder=0)

    # Plot spinodal line (no label - already in left panel)
    ax2.plot(beta_spinodal, h_spinodal, 'k--', linewidth=2.5, zorder=10)

    # Plot each model with individual colors/markers and error bars (matching plot2.py)
    for _, row in median_df.iterrows():
        short_name = row['short_name']
        color = MODEL_COLORS.get(short_name, '#333333')
        marker = MODEL_MARKERS.get(short_name, 'o')

        # Plot error bars (no label — legend built manually below)
        ax2.errorbar(row['beta'], row['h'],
                    xerr=[[row['beta_err_lower']], [row['beta_err_upper']]],
                    yerr=[[row['h_err_lower']], [row['h_err_upper']]],
                    fmt=marker, markersize=10, color=color,
                    markeredgecolor='white', markeredgewidth=0.5,
                    ecolor=color, elinewidth=2, capsize=4, capthick=2,
                    alpha=0.8, zorder=5)

    # Build legend with marker-only handles (no error bar lines)
    legend_handles = []
    legend_labels = []
    for _, row in median_df.iterrows():
        short_name = row['short_name']
        color = MODEL_COLORS.get(short_name, '#333333')
        marker = MODEL_MARKERS.get(short_name, 'o')
        handle = Line2D([0], [0], marker=marker, color=color, linestyle='None',
                        markersize=10, markeredgecolor='white', markeredgewidth=0.5)
        legend_handles.append(handle)
        legend_labels.append(short_name)

    ax2.legend(legend_handles, legend_labels,
               loc='upper right', fontsize=10, ncol=1, frameon=True, framealpha=0.9)

    ax2.set_xlabel(r'Majority force $\beta$', fontsize=16)
    ax2.set_ylabel(r'Bias magnitude $|h|$', fontsize=16)
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 1.5)
    # No legend - we have labels for each model

    # ===== RIGHT PANEL: Metastability scatter plot (like metastability_all_models.pdf) =====
    ax3 = axes[2]

    # Map metastability model names to short names (matching plot2.py)
    def get_meta_short_name(model_name):
        name_lower = model_name.lower()
        if 'llama-3.1-8b' in name_lower:
            return 'Llama-3.1-8B'
        elif 'gemma-3-27b' in name_lower:
            return 'Gemma-3-27B'
        elif 'gemma-3-12b' in name_lower:
            return 'Gemma-3-12B'
        elif 'gemini-2.5-flash' in name_lower:
            return 'Gemini-2.5-Flash'
        elif 'qwen3-32b' in name_lower:
            return 'Qwen3-32B'
        elif 'qwen3-14b' in name_lower:
            return 'Qwen3-14B'
        elif 'qwen2.5-32b' in name_lower:
            return 'Qwen2.5-32B'
        elif 'qwen2.5-14b' in name_lower:
            return 'Qwen2.5-14B'
        elif 'gpt-5-mini' in name_lower:
            return 'GPT-5-Mini'
        else:
            return model_name

    # Fill metastable region (no label - already in left panel)
    ax3.fill_between(beta_spinodal, 0, h_spinodal,
                    alpha=0.3, color='gray',
                    zorder=2)

    # Plot spinodal line - dashed style matching other panels (no label)
    ax3.plot(beta_spinodal, h_spinodal, 'k--', linewidth=2.5, zorder=10)

    # Plot each model separately with its marker from plot2.py
    models = sorted(metastability['model'].unique())
    for model_name in models:
        model_df = metastability[metastability['model'] == model_name]
        short_name = get_meta_short_name(model_name)
        color = MODEL_COLORS.get(short_name, '#333333')
        marker = MODEL_MARKERS.get(short_name, 'o')

        scatter = ax3.scatter(
            model_df['beta'],
            model_df['abs_h'],
            c=model_df['flip_fraction'],
            cmap='RdYlBu_r',
            marker=marker,
            s=80,
            edgecolors='black',
            linewidth=0.5,
            vmin=0,
            vmax=1,
            zorder=5,
            alpha=0.8,
            label=short_name
        )

    # Add colorbar in its dedicated column
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Flip fraction', fontsize=14)

    ax3.set_xlabel(r'Majority force $\beta$', fontsize=16)
    ax3.set_ylabel(r'Bias magnitude $|h|$', fontsize=16)
    ax3.set_xlim(0, 8)
    ax3.set_ylim(0, 1.5)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 1])

    # Save plots
    output_png = f'{BASE_PATH}/three_panel_phase_diagram.png'
    output_pdf = f'{BASE_PATH}/three_panel_phase_diagram.pdf'

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_png}")

    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    print(f"Plot (PDF) saved to: {output_pdf}")

    plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
