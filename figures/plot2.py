from pathlib import Path
"""
Two-panel figure showing:
1. Left panel: Gemma 3 27B with 3 opinion pairs demonstrating the role of h (bias)
2. Right panel: Data collapse plot with all 9 models at N=50

Uses results_batched_vllm folder data.
"""

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# Set random seed for reproducible shuffling
np.random.seed(42)

# General settings
FIG_DPI = 300
SCATTER_SIZE = 60
N_SIZE = 50

# Enhanced font setup
rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
})

def fit_func(m, beta, delta_m, delta_P):
    """Transition probability function: P(m) = 0.5 * tanh(beta * (m + delta_m)) + 0.5 + delta_P"""
    return 0.5 * np.tanh(beta * (m + delta_m)) + 0.5 + delta_P

# Define base path
BASE_PATH = str(Path(__file__).parent.parent / 'data')

# Define models and their configurations (9 models with N=50)
models_config = [
    {'name': 'Gemma-3-27B', 'path': f'{BASE_PATH}/gemma-3-27b-it/results_batched_vllm'},
    {'name': 'Gemma-3-12B', 'path': f'{BASE_PATH}/gemma-3-12b-it/results_batched_vllm'},
    {'name': 'Llama-3.1-8B', 'path': f'{BASE_PATH}/Llama-3.1-8B-Instruct/results_batched_vllm'},
    {'name': 'Qwen2.5-14B', 'path': f'{BASE_PATH}/Qwen2.5-14B-Instruct/results_batched_vllm'},
    {'name': 'Qwen2.5-32B', 'path': f'{BASE_PATH}/Qwen2.5-32B-Instruct/results_batched_vllm'},
    {'name': 'Qwen3-14B', 'path': f'{BASE_PATH}/Qwen3-14B/results_batched_vllm'},
    {'name': 'Qwen3-32B', 'path': f'{BASE_PATH}/Qwen3-32B/results_batched_vllm'},
    {'name': 'Gemini-2.5-Flash', 'path': f'{BASE_PATH}/gemini-2.5-flash-lite/results_batched_vllm'},
    {'name': 'GPT-5-Mini', 'path': f'{BASE_PATH}/gpt-5-mini/results_batched_vllm'},
]

# Color palette for models
model_colors = {
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

# Markers for models
model_markers = {
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

def extract_opinion_from_filename(filename):
    """Extract opinion name from filename.

    Handles two formats:
    - transition_prob_50_0.2_liberal.txt (format 1)
    - transition_prob_50_opinions_liberal.txt (format 2)
    """
    basename = os.path.basename(filename)
    # Remove prefix and extension
    name = basename.replace('.txt', '')

    # Format 2: transition_prob_50_opinions_<opinion>
    if '_opinions_' in name:
        parts = name.split('_opinions_')
        if len(parts) > 1:
            return parts[1].replace('_', ' ')

    # Format 1: transition_prob_50_0.2_<opinion>
    parts = name.split('_')
    # Opinion name starts after 'transition_prob_50_0.2_' (4 parts)
    if len(parts) > 4:
        opinion = '_'.join(parts[4:])
        return opinion.replace('_', ' ')

    return None

def process_model_data(model_name, base_path, opinion_pairs_df):
    """Process data for a single model at N=50.

    Only processes opinion pairs that are explicitly listed in opinion_pairs_df.
    """
    model_results = []

    file_pattern = f'{base_path}/N={N_SIZE}/transition_prob_*.txt'
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"  No files found for {model_name}")
        return model_results

    # Create set of valid opinion pairs from CSV for fast lookup
    valid_opinions_A = set(opinion_pairs_df['Opinion_A'].str.lower().str.strip())
    valid_opinions_B = set(opinion_pairs_df['Opinion_B'].str.lower().str.strip())
    valid_opinions = valid_opinions_A | valid_opinions_B

    for filepath in file_list:
        opinion1 = extract_opinion_from_filename(filepath)
        if opinion1 is None:
            continue

        # Check if this opinion is in the CSV file
        opinion1_lower = opinion1.lower().strip()
        if opinion1_lower not in valid_opinions:
            continue

        # Find matching opinion pair from CSV
        opinion2_row = opinion_pairs_df[opinion_pairs_df['Opinion_A'].str.lower().str.strip() == opinion1_lower]
        if opinion2_row.empty:
            # Try Opinion_B
            opinion2_row = opinion_pairs_df[opinion_pairs_df['Opinion_B'].str.lower().str.strip() == opinion1_lower]
            if not opinion2_row.empty:
                opinion2 = opinion2_row['Opinion_A'].values[0]
            else:
                continue
        else:
            opinion2 = opinion2_row['Opinion_B'].values[0]

        try:
            # Try to read data - handle both formats
            # Format 1: has header (m0, count_A, count_B, probability, standard_error)
            # Format 2: no header (m, col2, col3, probability, std_error)
            try:
                data = pd.read_csv(filepath)
                if 'm0' in data.columns:
                    x = data['m0'].astype(float).values
                    y = data['probability'].astype(float).values
                    if 'standard_error' in data.columns:
                        yerr = data['standard_error'].astype(float).values
                    elif 'count_A' in data.columns and 'count_B' in data.columns:
                        n = data['count_A'].astype(float).values + data['count_B'].astype(float).values
                        yerr = np.sqrt(y * (1 - y) / np.where(n > 0, n, 1))
                    else:
                        yerr = None
                else:
                    x = data.iloc[:, 0].astype(float).values
                    y = data.iloc[:, 3].astype(float).values
                    yerr = None
            except:
                # No header format
                data = pd.read_csv(filepath, header=None)
                x = data.iloc[:, 0].astype(float).values
                yerr = None
                if data.shape[1] >= 3:
                    col2 = data.iloc[:, 1].astype(float).values
                    col3 = data.iloc[:, 2].astype(float).values
                    denom = col2 + col3
                    y = np.where(denom > 0, col2 / denom, np.nan)
                    n = denom
                    y_safe = np.where(denom > 0, y, 0.5)
                    yerr = np.sqrt(y_safe * (1 - y_safe) / np.where(denom > 0, denom, 1))
                else:
                    continue

            # Remove NaN values
            valid_mask = np.isfinite(x) & np.isfinite(y)
            x = x[valid_mask]
            y = y[valid_mask]

            if len(x) < 4:
                continue

            # Bias correction: randomly shuffle 50% of opinion pairs
            shuffle_this_pair = np.random.random() < 0.5

            if shuffle_this_pair:
                opinion1, opinion2 = opinion2, opinion1
                x = -x
                y = 1 - y
                # SE is symmetric: sqrt(p(1-p)/n) = sqrt((1-p)p/n)

            # Fit the function
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                try:
                    popt, _ = curve_fit(
                        fit_func, x, y,
                        maxfev=5000,
                        p0=[2.0, 0.0, 0.0],
                        bounds=([-np.inf, -np.inf, -0.1], [np.inf, np.inf, 0.1])
                    )
                except (RuntimeError, OptimizeWarning):
                    continue

            beta, delta_m, delta_P = popt
            mse = np.mean((y - fit_func(x, *popt))**2)

            model_results.append({
                'Model': model_name,
                'Opinion_1': opinion1,
                'Opinion_2': opinion2,
                'beta': beta,
                'delta_m': delta_m,
                'delta_P': delta_P,
                'MSE': mse,
                'Filepath': filepath,
                'Shuffled': shuffle_this_pair,
                'x_data': x,
                'y_data': y,
                'yerr_data': yerr,
            })

        except Exception as e:
            continue

    print(f"  {model_name}: {len(model_results)} fits from opinion_pairs.csv")
    return model_results

def main():
    # Load opinion pairs
    opinion_pairs = pd.read_csv(f'{BASE_PATH}/opinion_pairs.csv')

    print("Processing models...")

    # Process all models
    all_fit_results = []
    for model_config in models_config:
        model_results = process_model_data(
            model_config['name'],
            model_config['path'],
            opinion_pairs
        )
        all_fit_results.extend(model_results)

    if not all_fit_results:
        print("No valid fits found!")
        return

    fit_df = pd.DataFrame(all_fit_results)
    print(f"\nTotal successful fits: {len(fit_df)}")

    # Create the figure with custom layout using gridspec
    fig = plt.figure(figsize=(16, 6), facecolor='white')
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1], hspace=0.35, wspace=0.35)

    # Left panel plot
    ax1 = fig.add_subplot(gs[0, 0])
    # Left panel legend area
    ax1_legend = fig.add_subplot(gs[1, 0])
    ax1_legend.axis('off')  # Hide the legend subplot axes

    # Right panel plot (spans both rows)
    ax2 = fig.add_subplot(gs[:, 1])

    # Set style for main plots (matching plot1.py)
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)

    # ===== LEFT PANEL: Gemma-3-27B with 3 opinion pairs showing role of h =====
    gemma_data = fit_df[fit_df['Model'] == 'Gemma-3-27B'].copy()

    if len(gemma_data) > 0:
        # Three pairs selected to illustrate the role of h at similar |h| magnitude:
        # 1. h ≈ 0:     "corporate personhood vs only humans have rights" (h=+0.007, β=3.5)
        # 2. h ≈ +0.33: "gender self-identification vs biological sex classification" (h=+0.325, β=2.9)
        # 3. h ≈ -0.32: "Israel support vs Palestine support" (h=-0.323, β=5.7)

        target_opinions = [
            ('corporate personhood', 'only humans have rights'),
            ('gender self-identification', 'biological sex classification'),
            ('Israel support', 'Palestine support'),
        ]

        example_rows = []
        for op1, op2 in target_opinions:
            # Find matching row (check both orderings due to shuffling)
            match = gemma_data[
                ((gemma_data['Opinion_1'].str.contains(op1, case=False, na=False)) &
                 (gemma_data['Opinion_2'].str.contains(op2, case=False, na=False))) |
                ((gemma_data['Opinion_1'].str.contains(op2, case=False, na=False)) &
                 (gemma_data['Opinion_2'].str.contains(op1, case=False, na=False)))
            ]
            if len(match) > 0:
                example_rows.append(match.iloc[0])
            else:
                # Fallback: find closest h to target
                print(f"Warning: Could not find {op1} vs {op2}")

        if len(example_rows) < 3:
            # Fallback to automatic selection if specific pairs not found
            target_h_values = [-0.5, 0.0, 0.5]
            for target_h in target_h_values:
                if len(example_rows) >= 3:
                    break
                closest_idx = (gemma_data['delta_m'] - target_h).abs().idxmin()
                example_rows.append(gemma_data.loc[closest_idx])

        # Use coolwarm colormap centered at h=0 (same as plot_simulation_results.py)
        cmap = plt.cm.coolwarm
        norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

        for i, row in enumerate(example_rows):
            x_data = row['x_data']
            y_data = row['y_data']

            op1 = row['Opinion_1']
            op2 = row['Opinion_2']
            h_val = row['delta_m']
            beta_val = row['beta']

            # Get color from colormap based on h value
            color = cmap(norm(h_val))

            # Plot data with binomial error bars
            yerr_data = row.get('yerr_data')
            ax1.errorbar(x_data, y_data, yerr=yerr_data,
                        fmt='o', color=color, markersize=6, alpha=0.9,
                        elinewidth=1.5, capsize=3, capthick=1.5,
                        markeredgecolor='white', markeredgewidth=1, zorder=5)

            # Plot fitted curve
            m_range = np.linspace(-1.1, 1.1, 200)
            P_fitted = fit_func(m_range, row['beta'], row['delta_m'], row['delta_P'])
            ax1.plot(m_range, P_fitted, color=color, linewidth=3, alpha=0.95, zorder=4,
                    label=f"{op1} vs {op2}")

        ax1.set_xlabel(r'Collective opinion $m$', fontsize=14)
        ax1.set_ylabel(r'Transition probability $P(m)$', fontsize=14)
        #ax1.set_title('Gemma 3 27B\nRole of bias h', fontsize=14, fontweight='bold', pad=10)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlim(-1.1, 1.1)

        # Place legend in the dedicated legend subplot below the plot
        legend_elements = ax1.get_legend_handles_labels()
        ax1_legend.legend(legend_elements[0], legend_elements[1], fontsize=12, loc='lower center',
                          frameon=True, fancybox=True, shadow=True, ncol=1)

        # Add colorbar for h values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax1, location='right', fraction=0.046, pad=0.04)
        cbar.set_label('Bias h', fontsize=14)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    # ===== RIGHT PANEL: Data collapse with all models =====
    # Create list of all data points with their model info
    all_collapse_points = []

    for _, row in fit_df.iterrows():
        x_data = row['x_data']
        y_data = row['y_data']
        beta = row['beta']
        delta_m = row['delta_m']
        delta_P = row['delta_P']

        # Rescale only the x axis; keep y as raw probability
        rescaled_m = beta * (x_data + delta_m)
        collapsed_P = y_data

        for rm, cp in zip(rescaled_m, collapsed_P):
            all_collapse_points.append({
                'Model': row['Model'],
                'Opinion': row['Opinion_1'],
                'rescaled_m': rm,
                'collapsed_P': cp
            })

    # Convert to DataFrame and shuffle to plot opinion by opinion (random order)
    collapse_df = pd.DataFrame(all_collapse_points)

    # Shuffle by opinion to ensure all models are visible
    unique_opinions = collapse_df['Opinion'].unique()
    np.random.shuffle(unique_opinions)

    # Plot data points opinion by opinion
    plotted_models = set()

    for opinion in unique_opinions:
        opinion_data = collapse_df[collapse_df['Opinion'] == opinion]
        for model in opinion_data['Model'].unique():
            model_data = opinion_data[opinion_data['Model'] == model]
            color = model_colors.get(model, '#333333')
            marker = model_markers.get(model, 'o')

            ax2.scatter(model_data['rescaled_m'], model_data['collapsed_P'],
                       s=35, alpha=0.5, color=color, marker=marker,
                       edgecolors='white', linewidth=0.3)

            plotted_models.add(model)

    # Plot universal curve: P = 0.5 * tanh(m*) + 0.5
    x_universal = np.linspace(-6, 6, 300)
    y_universal = 0.5 * np.tanh(x_universal) + 0.5
    ax2.plot(x_universal, y_universal, color='#2d3436', linestyle='--', linewidth=3,
             zorder=10, alpha=0.95, label='Universal curve')

    ax2.set_xlabel(r'Rescaled opinion $m^* = \beta(m + h)$', fontsize=14)
    ax2.set_ylabel(r'Transition probability $P(m^*)$', fontsize=14)
    #ax2.set_title('Data Collapse\nAll Models (N=50)', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(-6, 6)

    # Create legend for right panel
    legend_handles = []
    legend_labels = []

    for model in sorted(plotted_models):
        color = model_colors.get(model, '#333333')
        marker = model_markers.get(model, 'o')
        handle = Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
                       markersize=8, markeredgecolor='white', markeredgewidth=0.5,
                       linestyle='None')
        legend_handles.append(handle)
        legend_labels.append(model)

    # Add universal curve to legend
    universal_handle = Line2D([0], [0], color='#2d3436', linewidth=3, linestyle='--')
    legend_handles.append(universal_handle)
    legend_labels.append('tanh$(m^*)$')

    ax2.legend(legend_handles, legend_labels, fontsize=9, loc='lower right',
              frameon=True, fancybox=True, ncol=1)

    # Save plots
    output_path_png = f'{BASE_PATH}/two_panel_collapse_figure.png'
    output_path_pdf = f'{BASE_PATH}/two_panel_collapse_figure.pdf'

    plt.savefig(output_path_png, dpi=FIG_DPI, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to: {output_path_png}")

    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Plot (PDF) saved to: {output_path_pdf}")

    plt.close()

    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Total fits: {len(fit_df)}")
    print(f"Shuffled pairs: {fit_df['Shuffled'].sum()} ({fit_df['Shuffled'].mean()*100:.1f}%)")

    print(f"\nFits per model:")
    for model in sorted(fit_df['Model'].unique()):
        count = len(fit_df[fit_df['Model'] == model])
        print(f"  {model}: {count}")

    print(f"\nBias (h) distribution:")
    print(f"  Mean h: {fit_df['delta_m'].mean():.3f}")
    print(f"  Std h: {fit_df['delta_m'].std():.3f}")
    print(f"  Min h: {fit_df['delta_m'].min():.3f}")
    print(f"  Max h: {fit_df['delta_m'].max():.3f}")


if __name__ == "__main__":
    main()
