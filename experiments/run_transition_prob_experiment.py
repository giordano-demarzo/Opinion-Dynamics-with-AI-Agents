#!/usr/bin/env python3
"""
Configurable script to run transition probability experiments with vLLM.

Usage:
    python run_transition_prob_experiment.py --model Qwen/Qwen3-4B --temp 0.2 --N 10 --N_sim 100
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
DATA_DIR = Path(__file__).parent.parent / "data"

import argparse
from opinion_dynamics_vllm import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run transition probability experiment with vLLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., 'Qwen/Qwen3-4B', 'meta-llama/Llama-3.1-8B-Instruct')"
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
        default=0.9,
        help="GPU memory utilization (0.0-1.0)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
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
        "--N",
        type=int,
        default=10,
        help="Total number of agents in the network"
    )
    parser.add_argument(
        "--N-sim",
        type=int,
        default=100,
        help="Number of simulations per magnetization value"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for vLLM inference"
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

    args = parser.parse_args()

    # Print configuration
    print("="*70)
    print("TRANSITION PROBABILITY EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Model:                    {args.model}")
    print(f"Tensor parallel size:     {args.tensor_parallel_size}")
    print(f"GPU memory utilization:   {args.gpu_memory_utilization}")
    print(f"Max model length:         {args.max_model_len if args.max_model_len else 'auto'}")
    print(f"Opinion pairs file:       {args.opinion_pairs_file}")
    print(f"Population size (N):      {args.N}")
    print(f"Simulations per m0:       {args.N_sim}")
    print(f"Batch size:               {args.batch_size}")
    print(f"Temperature:              {args.temperature}")
    print(f"Max tokens:               {args.max_tokens}")
    print(f"Save results:             {not args.no_save}")
    print(f"Generate plots:           {not args.no_plots}")
    print("="*70)
    print()

    # Run experiment
    results = run_experiment(
        model_id=args.model,
        opinion_pairs_file=args.opinion_pairs_file,
        N=args.N,
        N_sim=args.N_sim,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        save_results=not args.no_save,
        generate_plots=not args.no_plots
    )

    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"Processed {len(results)} opinion pairs")

    if not args.no_save:
        model_name = args.model.split("/")[-1]
        results_dir = str(DATA_DIR / model_name / "results_batched_vllm" / f"N={args.N}")
        print(f"Results saved to: {results_dir}")

    print("="*70)


if __name__ == "__main__":
    main()
