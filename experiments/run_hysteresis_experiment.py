#!/usr/bin/env python3
"""
Run hysteresis cycle experiments for multiple opinion pairs.

This script loads opinion pairs from a CSV file and runs hysteresis simulations
for each pair, measuring how opinions change as external field (stubborn agents) varies.

The variable-size hysteresis adds/removes stubborn agents from -max_stubborn to +max_stubborn,
which has been shown to match spinodal theory predictions well.

Usage:
    python run_hysteresis_experiment.py --model google/gemma-3-27b-it --opinion-pairs opinion_pairs.csv

    # Only run pairs with |h| <= 0.15:
    python run_hysteresis_experiment.py --model google/gemma-3-27b-it --max-h 0.15
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
DATA_DIR = Path(__file__).parent.parent / "data"

import argparse
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
import warnings
from vllm import LLM, SamplingParams
from opinion_dynamics_vllm import simulate_hysteresis_cycle_direct_control, plot_hysteresis_results


def fit_func(m, beta, h, delta_P):
    """Transition probability function for fitting h."""
    return 0.5 * np.tanh(beta * (m + h)) + 0.5 + delta_P


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


def is_in_spinodal_region(beta, h):
    """
    Check if (beta, h) is in the spinodal (metastable) region.
    Returns True if beta > 1 and |h| <= h_spinodal(beta).
    """
    if beta <= 1:
        return False
    h_spinodal = find_spinodal_infinite(beta)
    if h_spinodal is None:
        return False
    return np.abs(h) <= h_spinodal


def get_h_values_for_opinion_pairs(model_path, opinion_pairs_df, N=50):
    """
    Fit beta and h from batched transition probability data for each opinion pair.

    Args:
        model_path: Path to model results (e.g., 'gemma-3-27b-it')
        opinion_pairs_df: DataFrame with Opinion_A, Opinion_B columns
        N: Number of agents used in batched experiments

    Returns:
        Dictionary mapping (opinion_A, opinion_B) to (beta, h) tuple
    """
    batched_path = DATA_DIR / model_path / "results_batched_vllm" / f"N={N}"

    if not batched_path.exists():
        print(f"Warning: Batched results not found at {batched_path}")
        return {}

    h_values = {}

    # Create lookup for valid opinions
    valid_opinions = set(opinion_pairs_df['Opinion_A'].str.lower().str.strip()) | \
                     set(opinion_pairs_df['Opinion_B'].str.lower().str.strip())

    for filepath in batched_path.glob("transition_prob_*.txt"):
        # Extract opinion from filename
        basename = filepath.stem
        parts = basename.split('_')
        if len(parts) > 4:
            opinion1 = '_'.join(parts[4:]).replace('_', ' ')
        else:
            continue

        opinion1_lower = opinion1.lower().strip()
        if opinion1_lower not in valid_opinions:
            continue

        # Find matching opinion pair
        match_A = opinion_pairs_df[opinion_pairs_df['Opinion_A'].str.lower().str.strip() == opinion1_lower]
        match_B = opinion_pairs_df[opinion_pairs_df['Opinion_B'].str.lower().str.strip() == opinion1_lower]

        if not match_A.empty:
            opinion_A = match_A['Opinion_A'].values[0]
            opinion_B = match_A['Opinion_B'].values[0]
        elif not match_B.empty:
            opinion_A = match_B['Opinion_A'].values[0]
            opinion_B = match_B['Opinion_B'].values[0]
        else:
            continue

        try:
            # Try reading with header first
            data = pd.read_csv(filepath)
            if 'm0' in data.columns:
                x = data['m0'].astype(float).values
                y = data['probability'].astype(float).values
            else:
                # No header, try reading without header
                data = pd.read_csv(filepath, header=None, names=['m0', 'count_A', 'count_B', 'probability', 'standard_error'])
                x = data['m0'].astype(float).values
                y = data['probability'].astype(float).values

            valid_mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid_mask], y[valid_mask]

            if len(x) < 4:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    fit_func, x, y,
                    maxfev=5000,
                    p0=[2.0, 0.0, 0.0],
                    bounds=([0.1, -1.0, -0.5], [20.0, 1.0, 0.5])
                )

            beta = popt[0]
            h = popt[1]
            h_values[(opinion_A, opinion_B)] = (beta, h)

        except Exception:
            continue

    return h_values


def detect_model_backend(model_name):
    """Detect which backend to use based on model name."""
    model_lower = model_name.lower()
    if 'gemini' in model_lower:
        return 'gemini'
    elif 'gpt' in model_lower or 'openai' in model_lower:
        return 'openai'
    else:
        return 'vllm'


def main():
    parser = argparse.ArgumentParser(
        description="Run hysteresis cycle experiments with vLLM, Gemini, or OpenAI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        help="HuggingFace model ID (vLLM), 'gemini-2.5-flash-lite' (Gemini), or 'gpt-5-mini' (OpenAI)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization (0.0-1.0)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model sequence length (reduces KV cache memory for large models)"
    )

    # Experiment parameters
    parser.add_argument(
        "--opinion-pairs-file",
        type=str,
        default="opinion_pairs.csv",
        help="CSV file with opinion pairs (columns: Opinion_A, Opinion_B)"
    )
    parser.add_argument(
        "--N-regular",
        type=int,
        default=50,
        help="Number of regular (changeable) agents"
    )
    parser.add_argument(
        "--max-stubborn",
        type=int,
        default=30,
        help="Maximum number of stubborn agents per side (sweep from -max to +max)"
    )
    parser.add_argument(
        "--n-cycles",
        type=int,
        default=1,
        help="Number of independent hysteresis cycles to average"
    )
    parser.add_argument(
        "--initial-equilibration",
        type=int,
        default=100,
        help="Equilibration steps for first field point"
    )
    parser.add_argument(
        "--step-equilibration",
        type=int,
        default=50,
        help="Equilibration steps for subsequent field points"
    )
    parser.add_argument(
        "--sampling-steps",
        type=int,
        default=25,
        help="Number of steps to sample magnetization at each field point"
    )

    # Generation parameters
    parser.add_argument(
        "--temp",
        "--temperature",
        type=float,
        default=0.2,
        dest="temperature",
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum tokens to generate"
    )

    # Output options
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Don't generate plots"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip opinion pairs that already have result files"
    )
    parser.add_argument(
        "--max-h",
        type=float,
        default=None,
        help="Only include opinion pairs with |h| <= max-h (requires batched results for fitting)"
    )
    parser.add_argument(
        "--spinodal-only",
        action="store_true",
        help="Only include opinion pairs within the spinodal (metastable) region (requires batched results)"
    )

    args = parser.parse_args()

    # Detect backend
    backend = detect_model_backend(args.model)

    # Print configuration
    print("="*70)
    print("HYSTERESIS CYCLE EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Backend:                  {backend.upper()}")
    print(f"Model:                    {args.model}")
    if backend == 'vllm':
        print(f"Tensor parallel size:     {args.tensor_parallel_size}")
        print(f"GPU memory utilization:   {args.gpu_memory_utilization}")
        print(f"Max model length:         {args.max_model_len}")
    print(f"Opinion pairs file:       {args.opinion_pairs_file}")
    print(f"Regular agents:           {args.N_regular}")
    print(f"Max stubborn agents:      {args.max_stubborn} (sweep from -{args.max_stubborn} to +{args.max_stubborn})")
    print(f"Number of cycles:         {args.n_cycles}")
    print(f"Initial equilibration:    {args.initial_equilibration}")
    print(f"Step equilibration:       {args.step_equilibration}")
    print(f"Sampling steps:           {args.sampling_steps}")
    print(f"Temperature:              {args.temperature}")
    print(f"Max tokens:               {args.max_tokens}")
    print(f"Save results:             {not args.no_save}")
    print(f"Generate plots:           {not args.no_plots}")
    print(f"Skip existing:            {args.skip_existing}")
    print(f"Max |h| filter:           {args.max_h if args.max_h else 'None (include all)'}")
    print(f"Spinodal region only:     {args.spinodal_only}")
    print("="*70)
    print()

    # Load opinion pairs
    if not Path(args.opinion_pairs_file).exists():
        print(f"Error: Opinion pairs file '{args.opinion_pairs_file}' not found!")
        print(f"Please create this file with columns: Opinion_A, Opinion_B")
        return

    # Read CSV file (comma-separated with header)
    try:
        opinion_pairs = pd.read_csv(args.opinion_pairs_file)
        # Verify required columns exist
        if 'Opinion_A' not in opinion_pairs.columns or 'Opinion_B' not in opinion_pairs.columns:
            raise ValueError("CSV must have 'Opinion_A' and 'Opinion_B' columns")
        # Remove empty rows
        opinion_pairs = opinion_pairs.dropna()
        print(f"Loaded {len(opinion_pairs)} opinion pairs from {args.opinion_pairs_file}")
    except Exception as e:
        print(f"Error reading opinion pairs file: {e}")
        return

    print()

    # Filter by h value and/or spinodal region if requested
    model_name = args.model.split("/")[-1]
    if args.max_h is not None or args.spinodal_only:
        # Determine what filters to apply
        filter_descriptions = []
        if args.max_h is not None:
            filter_descriptions.append(f"|h| <= {args.max_h}")
        if args.spinodal_only:
            filter_descriptions.append("spinodal region")

        print(f"Filtering opinion pairs by: {' AND '.join(filter_descriptions)}...")
        beta_h_values = get_h_values_for_opinion_pairs(model_name, opinion_pairs, N=args.N_regular)

        if not beta_h_values:
            print(f"Warning: Could not load beta/h values from batched results. Cannot apply filters.")
            print(f"Make sure batched results exist in {model_name}/results_batched_vllm/N={args.N_regular}/")
        else:
            # Filter opinion pairs
            filtered_rows = []
            excluded_h_count = 0
            excluded_spinodal_count = 0
            no_data_count = 0

            for _, row in opinion_pairs.iterrows():
                opinion_A = row['Opinion_A']
                opinion_B = row['Opinion_B']
                key = (opinion_A, opinion_B)

                if key in beta_h_values:
                    beta, h = beta_h_values[key]

                    # Check all filters
                    passes_max_h = (args.max_h is None) or (abs(h) <= args.max_h)
                    passes_spinodal = (not args.spinodal_only) or is_in_spinodal_region(beta, h)

                    if passes_max_h and passes_spinodal:
                        filtered_rows.append(row)
                        print(f"  ✓ {opinion_A} vs {opinion_B}: β={beta:.3f}, h={h:+.4f}")
                    else:
                        if not passes_max_h:
                            excluded_h_count += 1
                            print(f"  ✗ {opinion_A} vs {opinion_B}: β={beta:.3f}, h={h:+.4f} (|h| > {args.max_h})")
                        elif not passes_spinodal:
                            excluded_spinodal_count += 1
                            print(f"  ✗ {opinion_A} vs {opinion_B}: β={beta:.3f}, h={h:+.4f} (outside spinodal)")
                else:
                    no_data_count += 1
                    print(f"  ? {opinion_A} vs {opinion_B}: no beta/h data (excluded)")

            opinion_pairs = pd.DataFrame(filtered_rows)
            print(f"\nFiltered: {len(opinion_pairs)} pairs passed all filters")
            if excluded_h_count > 0:
                print(f"Excluded: {excluded_h_count} pairs with |h| > {args.max_h}")
            if excluded_spinodal_count > 0:
                print(f"Excluded: {excluded_spinodal_count} pairs outside spinodal region")
            if no_data_count > 0:
                print(f"No data: {no_data_count} pairs")
        print()

    # Setup results directory and check for existing files
    if backend == 'vllm':
        results_subdir = "results_hysteresis_vllm"
    elif backend == 'gemini':
        results_subdir = "results_hysteresis_gemini"
    else:  # openai
        results_subdir = "results_hysteresis_openai"

    results_dir = DATA_DIR / model_name / results_subdir / f"N={args.N_regular}"
    results_dir.mkdir(parents=True, exist_ok=True)

    existing_pairs = set()
    if args.skip_existing:
        if backend == 'vllm':
            existing_files = list(results_dir.glob(f"hysteresis_*_T{args.temperature:.2f}.txt"))
        else:
            existing_files = list(results_dir.glob(f"hysteresis_*.txt"))
        for f in existing_files:
            existing_pairs.add(f.stem)
        print(f"Found {len(existing_pairs)} existing result files")
        print()

    # Import appropriate module and initialize model
    llm = None
    sampling_params = None

    if backend == 'vllm':
        from vllm import LLM, SamplingParams
        from opinion_dynamics_vllm import simulate_hysteresis_cycle_direct_control, plot_hysteresis_results

        print("Initializing vLLM model...")
        llm_kwargs = {
            "model": args.model,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "trust_remote_code": True
        }
        if args.max_model_len is not None:
            llm_kwargs["max_model_len"] = args.max_model_len
            print(f"Setting max_model_len to {args.max_model_len} to reduce KV cache memory")

        llm = LLM(**llm_kwargs)

        # Define sampling parameters
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=0.95,
            max_tokens=args.max_tokens
        )
        print("Model initialized successfully!")

    elif backend == 'gemini':
        from opinion_dynamics_gemini_api import simulate_hysteresis_cycle_direct_control, plot_hysteresis_results
        print("Using Gemini API (no model initialization needed)")

    elif backend == 'openai':
        from opinion_dynamics_openai_api import simulate_hysteresis_cycle_direct_control, plot_hysteresis_results
        print("Using OpenAI API (no model initialization needed)")

    print()

    # Process each opinion pair
    all_results = []

    for idx, row in opinion_pairs.iterrows():
        opinion_A = row['Opinion_A']
        opinion_B = row['Opinion_B']

        # Check if already processed
        if backend == 'vllm':
            expected_filename = f"hysteresis_{opinion_A}_vs_{opinion_B}_T{args.temperature:.2f}"
        else:
            expected_filename = f"hysteresis_{opinion_A}_vs_{opinion_B}"

        if args.skip_existing and expected_filename in existing_pairs:
            print(f"\n[{idx+1}/{len(opinion_pairs)}] Skipping '{opinion_A}' vs '{opinion_B}' (already exists)")
            all_results.append({
                'opinion_A': opinion_A,
                'opinion_B': opinion_B,
                'status': 'skipped'
            })
            continue

        print(f"\n{'='*70}")
        print(f"Opinion pair {idx+1}/{len(opinion_pairs)}")
        print(f"Opinion A: '{opinion_A}'")
        print(f"Opinion B: '{opinion_B}'")
        print(f"{'='*70}")

        try:
            # Run hysteresis simulation with variable stubborn agent count
            if backend == 'vllm':
                sweep_data = simulate_hysteresis_cycle_direct_control(
                    llm=llm,
                    sampling_params=sampling_params,
                    opinion_A=opinion_A,
                    opinion_B=opinion_B,
                    N_regular=args.N_regular,
                    max_stubborn=args.max_stubborn,
                    initial_equilibration_steps=args.initial_equilibration,
                    step_equilibration_steps=args.step_equilibration,
                    sampling_steps=args.sampling_steps,
                    n_repeats=args.n_cycles,
                    model_id=args.model,
                    save_results=not args.no_save
                )
            elif backend == 'gemini':
                sweep_data = simulate_hysteresis_cycle_direct_control(
                    opinion_A=opinion_A,
                    opinion_B=opinion_B,
                    N_regular=args.N_regular,
                    max_stubborn=args.max_stubborn,
                    initial_equilibration_steps=args.initial_equilibration,
                    step_equilibration_steps=args.step_equilibration,
                    sampling_steps=args.sampling_steps,
                    n_repeats=args.n_cycles,
                    max_tokens=args.max_tokens,
                    model_id=args.model,
                    save_results=not args.no_save
                )
            elif backend == 'openai':
                sweep_data = simulate_hysteresis_cycle_direct_control(
                    opinion_A=opinion_A,
                    opinion_B=opinion_B,
                    N_regular=args.N_regular,
                    max_stubborn=args.max_stubborn,
                    initial_equilibration_steps=args.initial_equilibration,
                    step_equilibration_steps=args.step_equilibration,
                    sampling_steps=args.sampling_steps,
                    n_repeats=args.n_cycles,
                    max_tokens=args.max_tokens,
                    model_id=args.model,
                    save_results=not args.no_save
                )

            # Generate plot if requested
            if not args.no_plots:
                plot_path = results_dir / f"plot_{opinion_A}_vs_{opinion_B}.png"
                plot_hysteresis_results(sweep_data, save_path=str(plot_path))

            all_results.append({
                'opinion_A': opinion_A,
                'opinion_B': opinion_B,
                'status': 'success'
            })

            print(f"\n✓ Successfully completed: {opinion_A} vs {opinion_B}")

        except Exception as e:
            print(f"\n✗ Error processing {opinion_A} vs {opinion_B}: {e}")
            import traceback
            traceback.print_exc()

            all_results.append({
                'opinion_A': opinion_A,
                'opinion_B': opinion_B,
                'status': 'failed',
                'error': str(e)
            })
            continue

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED")
    print("="*70)

    successful = sum(1 for r in all_results if r['status'] == 'success')
    skipped = sum(1 for r in all_results if r['status'] == 'skipped')
    failed = sum(1 for r in all_results if r['status'] == 'failed')

    print(f"\nTotal opinion pairs: {len(all_results)}")
    print(f"Successful:          {successful}")
    print(f"Skipped (existing):  {skipped}")
    print(f"Failed:              {failed}")

    if successful > 0:
        print(f"\nSuccessful opinion pairs:")
        for r in all_results:
            if r['status'] == 'success':
                print(f"  ✓ {r['opinion_A']} vs {r['opinion_B']}")

    if failed > 0:
        print(f"\nFailed opinion pairs:")
        for r in all_results:
            if r['status'] == 'failed':
                print(f"  ✗ {r['opinion_A']} vs {r['opinion_B']}")

    if not args.no_save:
        print(f"\nResults saved to: {results_dir}")

    print("="*70)


if __name__ == "__main__":
    main()
