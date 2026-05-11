"""
Opinion Dynamics Simulation using vLLM Batch Inference

This script implements three main functionalities:
1. Single agent simulation (sequential opinion evolution)
2. Full opinion dynamics on fully connected network (transition probabilities)
3. Hysteresis cycle generation with external field

Converted from opinion_dynamics_batches_v2.ipynb to use vLLM for efficient batch inference.
"""

import os
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from vllm import LLM, SamplingParams


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_unique_random_strings(length: int = 2, num_strings: int = 10) -> List[str]:
    """
    Generate a list of unique random alphanumeric strings.

    Args:
        length: Length of each string (default: 2)
        num_strings: Number of unique strings to generate

    Returns:
        List of unique random strings
    """
    characters = string.ascii_letters + string.digits
    strings_set = set()

    while len(strings_set) < num_strings:
        random_string = ''.join(random.choices(characters, k=length))
        strings_set.add(random_string)

    return list(strings_set)


def synchronized_shuffle(list_a: List, list_b: List) -> Tuple[List, List]:
    """
    Shuffle two lists with the same random permutation.

    Args:
        list_a: First list to shuffle
        list_b: Second list to shuffle (must be same length as list_a)

    Returns:
        Tuple of (shuffled_list_a, shuffled_list_b)
    """
    combined = list(zip(list_a, list_b))
    random.shuffle(combined)
    shuffled_a, shuffled_b = zip(*combined)
    return list(shuffled_a), list(shuffled_b)


def apply_chat_template_to_prompts(llm: LLM, prompts: List[str], debug: bool = False) -> List[str]:
    """
    Apply chat template to prompts using the model's tokenizer.

    Args:
        llm: vLLM LLM instance
        prompts: List of raw prompt strings
        debug: If True, print debug information

    Returns:
        List of formatted prompts with chat template applied
    """
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    
    # Detect model type from tokenizer
    model_name = getattr(tokenizer, 'name_or_path', '').lower()
    is_qwen = 'qwen' in model_name
    is_gemma = 'gemma' in model_name

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        
        template_method = None
        try:
            # Only use enable_thinking parameter for Qwen models
            if is_qwen:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                template_method = "qwen_chat_template"
            else:
                # For all other models (including Gemma), use standard template
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                template_method = "standard_chat_template"
                
        except (ValueError, AttributeError):
            # Fallback if chat template is not set
            formatted = f"[INST] {prompt} [/INST]"
            template_method = "fallback_template"

        if debug and len(formatted_prompts) == 0:
            print(f"[DEBUG] Model type: Qwen={is_qwen}, Gemma={is_gemma}")
            print(f"[DEBUG] Chat template method: {template_method}")

        formatted_prompts.append(formatted)

    return formatted_prompts


def parse_opinion_response(response: str, opinion_A: str, opinion_B: str) -> Optional[str]:
    """
    Parse the LLM response to extract the chosen opinion.

    Expected format: [opinion]

    Args:
        response: Raw LLM response text
        opinion_A: First opinion option
        opinion_B: Second opinion option

    Returns:
        The chosen opinion (opinion_A or opinion_B) or None if invalid
    """
    if "[" in response and "]" in response:
        chosen_opinion = response.partition("[")[2].partition("]")[0].strip()
        if chosen_opinion == opinion_A:
            return opinion_A
        elif chosen_opinion == opinion_B:
            return opinion_B
    return None


def create_opinion_prompt(agent_names: List[str],
                         agent_opinions: List[str],
                         opinion_A: str,
                         opinion_B: str) -> str:
    """
    Create a prompt showing agents and their opinions.

    Args:
        agent_names: List of agent names
        agent_opinions: List of agent opinions (same length as agent_names)
        opinion_A: First opinion option
        opinion_B: Second opinion option

    Returns:
        Formatted prompt string
    """
    # Randomly order the available opinions to prevent order bias
    available_opinions = [opinion_A, opinion_B]
    random.shuffle(available_opinions)

    prompt = (
        "Below you can see the list of all the other AI assitants with the opinion they support.\n"
        f"You must reply with the opinion you want to support.\n"
        #f"You must reply with the opinion you want to support, either {available_opinions[0]} or {available_opinions[1]}.\n"
        #f"The available opinions are {available_opinions[0]} and {available_opinions[1]}.\n"
        "The opinion must be reported between square brackets.\n"
    )

    for name, opinion in zip(agent_names, agent_opinions):
        prompt += f"[{name}]: {opinion}\n"

    prompt += "Reply only with the opinion you want to support, between square brackets."

    return prompt


# =============================================================================
# MULTI AGENTS SIMULATION
# =============================================================================

def simulate_opinion_dynamics(
    llm: LLM,
    sampling_params: SamplingParams,
    Na_init: int,
    N: int,
    t_max: int,
    opinion1: str,
    opinion2: str,
    model_id: str,
    sim_id: int = 0,
    save_results: bool = True
) -> List[Tuple[int, float]]:
    """
    Simulate sequential opinion evolution on a fully connected network.

    Each time step, one randomly selected agent observes all other agents
    and updates their opinion based on the social environment.

    Args:
        llm: vLLM LLM instance
        sampling_params: vLLM SamplingParams for generation
        Na_init: Initial number of agents supporting opinion A
        N: Total number of agents in the network
        t_max: Maximum time steps to simulate
        opinion1: First opinion (opinion A)
        opinion2: Second opinion (opinion B)
        model_id: Model identifier for file naming
        sim_id: Simulation ID for file naming
        save_results: Whether to save magnetization to file

    Returns:
        List of (timestep, magnetization) tuples
    """
    print(f"\n{'='*60}")
    print(f"Starting Single Agent Simulation")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Population: N={N}")
    print(f"Initial state: Na={Na_init}, Nb={N-Na_init}")
    print(f"Max steps: {t_max}")
    print(f"Opinions: '{opinion1}' vs '{opinion2}'")

    # Initialize agent counts
    Na = Na_init
    Nb = N - Na

    # Track magnetization over time
    magnetization_history = []

    # Initial magnetization
    m_initial = (2 * Na - N) / N
    m = m_initial
    magnetization_history.append((0, m))
    initial_sign = np.sign(m_initial)
    print(f"\nInitial magnetization: m = {m:.4f} (sign: {'+' if initial_sign > 0 else '-'})")

    # Main simulation loop
    for t in range(1, t_max + 1):
        # Check for consensus
        if abs(m) == 1.0:
            print(f"\nConsensus reached at t={t-1} with m={m:.4f}")
            break

        # Generate unique agent names
        agent_names = generate_unique_random_strings(length=2, num_strings=N)

        # Create opinion list: Na agents with opinion1, Nb with opinion2
        opinions = [opinion1] * Na + [opinion2] * Nb

        # Shuffle names and opinions together
        agent_names, opinions = synchronized_shuffle(agent_names, opinions)

        # Randomly select one agent to update
        idx = random.randint(0, N - 1)
        current_opinion = opinions[idx]

        # Create list of OTHER agents (excluding the selected agent)
        other_names = agent_names[:idx] + agent_names[idx+1:]
        other_opinions = opinions[:idx] + opinions[idx+1:]

        # Create prompt showing social environment
        prompt = create_opinion_prompt(other_names, other_opinions, opinion1, opinion2)

        # Get LLM response (apply chat template for proper behavior)
        formatted_prompts = apply_chat_template_to_prompts(llm, [prompt], debug=(t <= 1))
        outputs = llm.generate(formatted_prompts, sampling_params)
        response = outputs[0].outputs[0].text

        # Debug: Print prompt and response for first few steps
        if t <= 3:
            print(f"\n--- DEBUG Step {t} ---")
            print(f"Selected agent index: {idx}, Current opinion: {current_opinion}")
            print(f"Raw prompt (first 500 chars):\n{prompt[:500]}...")
            print(f"Formatted prompt (first 700 chars):\n{formatted_prompts[0][:700]}...")
            print(f"Response: '{response}'")

        # Parse response
        chosen_opinion = parse_opinion_response(response, opinion1, opinion2)

        # Debug: Print parsed opinion
        if t <= 3:
            print(f"Parsed opinion: {chosen_opinion}")
            print(f"Opinion changed: {chosen_opinion != current_opinion if chosen_opinion else 'Invalid response'}")
            print("--- END DEBUG ---\n")

        # Update counts if opinion changed
        if chosen_opinion is not None and chosen_opinion != current_opinion:
            if chosen_opinion == opinion1:
                Na += 1
                Nb -= 1
            else:
                Na -= 1
                Nb += 1

            # Update magnetization
            m = (2 * Na - N) / N

        # Record magnetization
        magnetization_history.append((t, m))

        # Check for early stopping: magnetization flip detected
        current_sign = np.sign(m)
        if current_sign != initial_sign and abs(m) > 0.99:
            print(f"\n🎯 Early stop at t={t}: Magnetization flipped!")
            print(f"   Initial: m={m_initial:.4f} (sign: {'+' if initial_sign > 0 else '-'})")
            print(f"   Current: m={m:.4f} (sign: {'+' if current_sign > 0 else '-'})")
            print(f"   Final state: Na={Na}, Nb={Nb}")
            break

        # Print progress every 10 steps
        if t % 10 == 0:
            print(f"Step {t}: Na={Na}, Nb={Nb}, m={m:.4f}")

    print(f"\nSimulation complete!")
    print(f"Final state: Na={Na}, Nb={Nb}, m={m:.4f}")

    # Save results to file
    if save_results:
        model_name = model_id.split("/")[-1]
        results_dir = Path(model_name) / "results_simulation_vllm" / f"N={N}"
        results_dir.mkdir(parents=True, exist_ok=True)

        T = sampling_params.temperature
        m0 = (2 * Na_init - N) / N
        filename = f"magnetization_{sim_id}_{N}_{T:.2f}_{m0:.2f}_{opinion1}_explicit_opinion.txt"
        filepath = results_dir / filename

        with open(filepath, 'w') as f:
            f.write("step,magnetization\n")
            for step, mag in magnetization_history:
                f.write(f"{step},{mag:.6f}\n")

        print(f"Results saved to: {filepath}")

    return magnetization_history


# =============================================================================
# INDIVIDUAL AGENT BIAS MEASUREMENT
# =============================================================================

def create_individual_bias_prompt(opinion_A: str, opinion_B: str) -> str:
    """
    Create a prompt for measuring individual agent bias without social context.

    The two opinions are presented in random order to eliminate position bias.

    Args:
        opinion_A: First opinion option
        opinion_B: Second opinion option

    Returns:
        Formatted prompt string
    """
    # Randomly order the opinions to prevent position bias
    opinions = [opinion_A, opinion_B]
    random.shuffle(opinions)

    prompt = (
        "Below you can see a list of opinions.\n"
        "You must reply with the opinion you want to support.\n"
        "The opinion must be reported between square brackets.\n"
    )

    for opinion in opinions:
        prompt += f"{opinion}\n"

    prompt += "Reply only with the opinion you want to support, between square brackets."

    return prompt


def measure_individual_bias(
    model_id: str,
    opinion_pairs_file: str = "opinion_pairs.csv",
    N_sim: int = 100,
    batch_size: int = 64,
    temperature: float = 0.2,
    max_new_tokens: int = 16,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    save_results: bool = True,
    generate_plots: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Measure individual agent bias for opinion pairs without social interaction.

    This experiment isolates the intrinsic bias of individual agents by presenting
    only the two opinions without any social context. Each agent independently
    chooses between the two opinions.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-32B-Instruct")
        opinion_pairs_file: CSV file with opinion pairs (columns: Opinion_A, Opinion_B)
        N_sim: Number of simulations (independent agents) per opinion pair
        batch_size: Batch size for vLLM inference
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        max_model_len: Maximum model sequence length (reduces KV cache memory usage if set)
        save_results: Whether to save results to files
        generate_plots: Whether to generate plots

    Returns:
        Dictionary mapping opinion pairs to DataFrames with bias measurements
    """
    print(f"\n{'='*60}")
    print(f"Starting Individual Agent Bias Measurement")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Independent agents per pair: N_sim={N_sim}")
    print(f"Temperature: {temperature}")
    print(f"Batch size: {batch_size}")

    # Initialize vLLM model
    print("\nInitializing vLLM model...")
    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
        print(f"Setting max_model_len to {max_model_len} to reduce KV cache memory")

    llm = LLM(**llm_kwargs)

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_new_tokens
    )

    # Warmup
    print("Performing warmup...")
    warmup_prompts = ["Test prompt" for _ in range(3)]
    warmup_formatted = apply_chat_template_to_prompts(llm, warmup_prompts)
    _ = llm.generate(warmup_formatted, sampling_params)
    print("Warmup complete!")

    # Load opinion pairs
    opinion_pairs = pd.read_csv(opinion_pairs_file)
    print(f"\nLoaded {len(opinion_pairs)} opinion pairs")

    # Store results
    all_results = {}

    # Process each opinion pair
    for idx, row in opinion_pairs.iterrows():
        opinion_A = row['Opinion_A']
        opinion_B = row['Opinion_B']

        print(f"\n{'='*60}")
        print(f"Opinion pair {idx+1}/{len(opinion_pairs)}")
        print(f"Opinion A: '{opinion_A}'")
        print(f"Opinion B: '{opinion_B}'")
        print(f"{'='*60}")

        # Keep generating batches until we get N_sim valid responses
        count_A = 0
        count_B = 0
        invalid_responses = []
        total_attempts = 0
        batch_num = 0
        max_attempts = N_sim * 10  # Safety limit: max 10x the required valid responses

        while count_A + count_B < N_sim and total_attempts < max_attempts:
            batch_num += 1
            # Calculate how many more valid responses we need
            remaining = N_sim - (count_A + count_B)

            # Generate batch of prompts
            batch_prompts = min(batch_size, remaining) if batch_num == 1 else min(batch_size, remaining * 2)
            prompts = []

            for sim in range(batch_prompts):
                # Create prompt (no social context, just two opinions)
                prompt = create_individual_bias_prompt(opinion_A, opinion_B)
                prompts.append(prompt)

            # Batch inference
            print(f"  Batch {batch_num}: Generating {len(prompts)} responses (need {remaining} more valid)...")

            # Apply chat template to all prompts for proper model behavior
            formatted_prompts = apply_chat_template_to_prompts(llm, prompts)
            outputs = llm.generate(formatted_prompts, sampling_params)

            # Count opinions from this batch
            batch_count_A = 0
            batch_count_B = 0
            batch_invalid = 0

            for i, output in enumerate(outputs):
                # Stop if we already have enough valid responses
                if count_A + count_B >= N_sim:
                    break

                # Extract text from RequestOutput
                response = output.outputs[0].text
                chosen_opinion = parse_opinion_response(response, opinion_A, opinion_B)

                if chosen_opinion == opinion_A:
                    count_A += 1
                    batch_count_A += 1
                elif chosen_opinion == opinion_B:
                    count_B += 1
                    batch_count_B += 1
                else:
                    batch_invalid += 1
                    # Save first 5 invalid responses for debugging
                    if len(invalid_responses) < 5:
                        invalid_responses.append(response)

            total_attempts += len(prompts)
            valid_in_batch = batch_count_A + batch_count_B
            print(f"    Batch results: A={batch_count_A}, B={batch_count_B}, invalid={batch_invalid}, "
                  f"total valid={count_A + count_B}/{N_sim}")

        # Calculate probability and standard error based on valid responses only
        valid_total = count_A + count_B

        # Check if we hit the maximum attempts limit
        if total_attempts >= max_attempts and valid_total < N_sim:
            print(f"  ⚠️  WARNING: Hit maximum attempts limit ({max_attempts}) with only {valid_total}/{N_sim} valid responses!")
            print(f"      This indicates the model is producing mostly invalid responses for this configuration.")
            print(f"      Results for this opinion pair will be based on {valid_total} samples instead of {N_sim}.")

        p = count_A / valid_total if valid_total > 0 else 0
        se = np.sqrt(p * (1 - p) / valid_total) if valid_total > 0 else 0

        invalid_count = total_attempts - valid_total
        print(f"  Final results: A={count_A}, B={count_B}, valid={valid_total}/{total_attempts}, "
              f"invalid={invalid_count}, p={p:.4f}±{se:.4f}")

        # Print sample invalid responses for debugging
        if invalid_responses:
            print(f"  Sample invalid responses ({len(invalid_responses)} shown):")
            for resp in invalid_responses:
                print(f"    '{resp[:80]}...'" if len(resp) > 80 else f"    '{resp}'")

        # Store results
        result_data = {
            'opinion_A': opinion_A,
            'opinion_B': opinion_B,
            'count_A': count_A,
            'count_B': count_B,
            'total_valid': valid_total,
            'probability_A': p,
            'standard_error': se
        }

        df = pd.DataFrame([result_data])
        all_results[f"{opinion_A}_vs_{opinion_B}"] = df

        # Save results
        if save_results:
            model_name = model_id.split("/")[-1]
            results_dir = Path(model_name) / "results_individual_bias_vllm"
            results_dir.mkdir(parents=True, exist_ok=True)

            filename = f"individual_bias_{temperature}_{opinion_A}.txt"
            filepath = results_dir / filename

            with open(filepath, 'w') as f:
                f.write("opinion_A,opinion_B,count_A,count_B,total_valid,probability_A,standard_error\n")
                f.write(f"{opinion_A},{opinion_B},{count_A},{count_B},{valid_total},"
                       f"{p:.6f},{se:.6f}\n")

            print(f"\nResults saved to: {filepath}")

    # Generate summary plot if requested
    if generate_plots and len(all_results) > 0:
        model_name = model_id.split("/")[-1]
        results_dir = Path(model_name) / "results_individual_bias_vllm"

        # Collect all probabilities
        opinion_labels = []
        probabilities = []
        errors = []

        for key, df in all_results.items():
            opinion_labels.append(df['opinion_A'].iloc[0])
            probabilities.append(df['probability_A'].iloc[0])
            errors.append(df['standard_error'].iloc[0])

        # Create bar plot
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(opinion_labels))
        plt.bar(x_pos, probabilities, yerr=errors, capsize=5, alpha=0.7, color='steelblue')
        plt.axhline(y=0.5, color='red', linestyle='--', label='No bias (0.5)')
        plt.xlabel('Opinion A', fontsize=12)
        plt.ylabel('P(choose Opinion A)', fontsize=12)
        plt.title(f'Individual Agent Bias - {model_name}', fontsize=14)
        plt.xticks(x_pos, opinion_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        plot_path = results_dir / f"bias_summary_T{temperature:.2f}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"\nSummary plot saved to: {plot_path}")
        plt.close()

    print(f"\n{'='*60}")
    print("Individual bias measurement complete!")
    print(f"{'='*60}")

    return all_results


# =============================================================================
# TRANSITION PROBABILITIES
# =============================================================================

def run_experiment(
    model_id: str,
    opinion_pairs_file: str = "opinion_pairs.csv",
    N: int = 10,
    N_sim: int = 100,
    batch_size: int = 64,
    temperature: float = 0.2,
    max_new_tokens: int = 16,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    save_results: bool = True,
    generate_plots: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Measure transition probabilities across different initial magnetizations
    using vLLM batch inference.

    For each opinion pair and each initial magnetization value, simulates
    N_sim independent trials to measure the probability that an agent
    adopts opinion A given the social environment.

    Args:
        model_id: HuggingFace model ID (e.g., "Qwen/Qwen3-32B-Instruct")
        opinion_pairs_file: CSV file with opinion pairs (columns: Opinion_A, Opinion_B)
        N: Total number of agents in the network
        N_sim: Number of simulations per magnetization value
        batch_size: Batch size for vLLM inference
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        max_model_len: Maximum model sequence length (reduces KV cache memory usage if set)
        save_results: Whether to save results to files
        generate_plots: Whether to generate plots

    Returns:
        Dictionary mapping opinion pairs to DataFrames with transition probabilities
    """
    print(f"\n{'='*60}")
    print(f"Starting Full Opinion Dynamics Experiment")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Population: N={N}")
    print(f"Simulations per magnetization: N_sim={N_sim}")
    print(f"Temperature: {temperature}")
    print(f"Batch size: {batch_size}")

    # Initialize vLLM model
    print("\nInitializing vLLM model...")
    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
        print(f"Setting max_model_len to {max_model_len} to reduce KV cache memory")

    llm = LLM(**llm_kwargs)

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_new_tokens
    )

    # Warmup
    print("Performing warmup...")
    warmup_prompts = ["Test prompt" for _ in range(3)]
    warmup_formatted = apply_chat_template_to_prompts(llm, warmup_prompts)
    _ = llm.generate(warmup_formatted, sampling_params)
    print("Warmup complete!")

    # Calculate magnetization grid
    # Based on: list_N = (N/50) * [50, 48, 45, 40, 30, 27, 23, 20, 10, 5, 2, 0]
    #base_values = [50, 48, 45, 40, 30, 27, 23, 20, 10, 5, 2, 0]
    base_values = [48, 45, 40, 30, 27, 23, 20, 10, 5, 2]
    list_N = [int(round((N / 50) * val)) for val in base_values]
    list_N = sorted(set(list_N), reverse=True)  # Remove duplicates and sort

    print(f"\nMagnetization grid: Na values = {list_N}")

    # Load opinion pairs
    opinion_pairs = pd.read_csv(opinion_pairs_file)
    print(f"\nLoaded {len(opinion_pairs)} opinion pairs")

    # Store results
    all_results = {}

    # Process each opinion pair
    for idx, row in opinion_pairs.iterrows():
        opinion_A = row['Opinion_A']
        opinion_B = row['Opinion_B']

        print(f"\n{'='*60}")
        print(f"Opinion pair {idx+1}/{len(opinion_pairs)}")
        print(f"Opinion A: '{opinion_A}'")
        print(f"Opinion B: '{opinion_B}'")
        print(f"{'='*60}")

        results_list = []

        # Process each magnetization value
        for Na in list_N:
            Nb = N - Na
            m0 = (Na - Nb) / N

            print(f"\nProcessing m0={m0:.4f} (Na={Na}, Nb={Nb})...")

            # Keep generating batches until we get N_sim valid responses
            count_A = 0
            count_B = 0
            invalid_responses = []
            total_attempts = 0
            batch_num = 0
            max_attempts = N_sim * 10  # Safety limit: max 10x the required valid responses

            while count_A + count_B < N_sim and total_attempts < max_attempts:
                batch_num += 1
                # Calculate how many more valid responses we need
                remaining = N_sim - (count_A + count_B)

                # Generate batch of prompts
                # Use batch_size for efficiency, but at least generate what we need
                batch_prompts = min(batch_size, remaining) if batch_num == 1 else min(batch_size, remaining * 2)
                prompts = []

                for sim in range(batch_prompts):
                    # Generate unique agent names
                    agent_names = generate_unique_random_strings(length=2, num_strings=N)

                    # Create opinion list
                    opinions = [opinion_A] * Na + [opinion_B] * Nb

                    # Shuffle
                    agent_names, opinions = synchronized_shuffle(agent_names, opinions)

                    # Create prompt (agent sees all N agents)
                    prompt = create_opinion_prompt(agent_names, opinions, opinion_A, opinion_B)
                    prompts.append(prompt)

                # Batch inference
                print(f"  Batch {batch_num}: Generating {len(prompts)} responses (need {remaining} more valid)...")

                # Apply chat template to all prompts for proper model behavior
                formatted_prompts = apply_chat_template_to_prompts(llm, prompts)
                outputs = llm.generate(formatted_prompts, sampling_params)

                # Count opinions from this batch
                batch_count_A = 0
                batch_count_B = 0
                batch_invalid = 0

                for i, output in enumerate(outputs):
                    # Stop if we already have enough valid responses
                    if count_A + count_B >= N_sim:
                        break

                    # Extract text from RequestOutput
                    response = output.outputs[0].text
                    chosen_opinion = parse_opinion_response(response, opinion_A, opinion_B)

                    if chosen_opinion == opinion_A:
                        count_A += 1
                        batch_count_A += 1
                    elif chosen_opinion == opinion_B:
                        count_B += 1
                        batch_count_B += 1
                    else:
                        batch_invalid += 1
                        # Save first 5 invalid responses for debugging
                        if len(invalid_responses) < 5:
                            invalid_responses.append(response)

                total_attempts += len(prompts)
                valid_in_batch = batch_count_A + batch_count_B
                print(f"    Batch results: A={batch_count_A}, B={batch_count_B}, invalid={batch_invalid}, "
                      f"total valid={count_A + count_B}/{N_sim}")

            # Calculate probability and standard error based on valid responses only
            valid_total = count_A + count_B

            # Check if we hit the maximum attempts limit
            if total_attempts >= max_attempts and valid_total < N_sim:
                print(f"  ⚠️  WARNING: Hit maximum attempts limit ({max_attempts}) with only {valid_total}/{N_sim} valid responses!")
                print(f"      This indicates the model is producing mostly invalid responses for this configuration.")
                print(f"      Results for this m0 will be based on {valid_total} samples instead of {N_sim}.")

            p = count_A / valid_total if valid_total > 0 else 0
            se = np.sqrt(p * (1 - p) / valid_total) if valid_total > 0 else 0

            invalid_count = total_attempts - valid_total
            print(f"  Final results: A={count_A}, B={count_B}, valid={valid_total}/{total_attempts}, "
                  f"invalid={invalid_count}, p={p:.4f}±{se:.4f}")

            # Print sample invalid responses for debugging
            if invalid_responses:
                print(f"  Sample invalid responses ({len(invalid_responses)} shown):")
                for resp in invalid_responses:
                    print(f"    '{resp[:80]}...'" if len(resp) > 80 else f"    '{resp}'")

            results_list.append({
                'm0': m0,
                'count_A': count_A,
                'count_B': count_B,
                'probability': p,
                'standard_error': se
            })

        # Convert to DataFrame
        df = pd.DataFrame(results_list)
        all_results[f"{opinion_A}_vs_{opinion_B}"] = df

        # Save results
        if save_results:
            model_name = model_id.split("/")[-1]
            results_dir = Path(model_name) / "results_batched_vllm_explicit_v2" / f"N={N}"
            results_dir.mkdir(parents=True, exist_ok=True)

            filename = f"transition_prob_{N}_{temperature}_{opinion_A}.txt"
            filepath = results_dir / filename

            with open(filepath, 'w') as f:
                f.write("m0,count_A,count_B,probability,standard_error\n")
                for _, row in df.iterrows():
                    f.write(f"{row['m0']:.6f},{row['count_A']},{row['count_B']},"
                           f"{row['probability']:.6f},{row['standard_error']:.6f}\n")

            print(f"\nResults saved to: {filepath}")

        # Generate plot
        if generate_plots:
            plt.figure(figsize=(8, 6))
            plt.errorbar(df['m0'], df['probability'], yerr=df['standard_error'],
                        fmt='o-', capsize=5, label=f'{opinion_A} vs {opinion_B}')
            plt.xlabel('Initial Magnetization m₀', fontsize=12)
            plt.ylabel(f'P(choose {opinion_A})', fontsize=12)
            plt.title(f'Transition Probability: {opinion_A} vs {opinion_B}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()

            if save_results:
                plot_path = results_dir / f"plot_{opinion_A}_vs_{opinion_B}.png"
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                print(f"Plot saved to: {plot_path}")

            plt.close()

    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"{'='*60}")

    return all_results


# =============================================================================
# HYSTERESIS CYCLE GENERATION
# =============================================================================

def simulate_hysteresis_step_with_state(
    llm: LLM,
    sampling_params: SamplingParams,
    current_regular_opinions: List[str],
    N_stubborn_A: int,
    N_stubborn_B: int,
    opinion_A: str,
    opinion_B: str,
    equilibration_steps: int,
    sampling_steps: int
) -> Tuple[float, float, List[str]]:
    """
    Simulate one field point in the hysteresis cycle.

    CRITICAL: This function modifies current_regular_opinions IN PLACE,
    maintaining state continuity between field points.

    Args:
        llm: vLLM LLM instance
        sampling_params: vLLM SamplingParams for generation
        current_regular_opinions: List of current opinions for regular agents (modified in place!)
        N_stubborn_A: Number of stubborn agents supporting opinion A
        N_stubborn_B: Number of stubborn agents supporting opinion B
        opinion_A: First opinion
        opinion_B: Second opinion
        equilibration_steps: Number of equilibration steps before sampling
        sampling_steps: Number of sampling steps to measure magnetization

    Returns:
        Tuple of (average_magnetization, std_magnetization, updated_opinions)
    """
    N_regular = len(current_regular_opinions)
    N_total = N_regular + N_stubborn_A + N_stubborn_B

    # Equilibration phase (updates not recorded)
    for _ in range(equilibration_steps):
        # Randomly select a regular agent
        idx = random.randint(0, N_regular - 1)

        # Generate unique names for all agents
        all_names = generate_unique_random_strings(length=2, num_strings=N_total)

        # Construct full opinion list: regular + stubborn
        all_opinions = (
            current_regular_opinions[:idx] +
            current_regular_opinions[idx+1:] +  # Other regular agents
            [opinion_A] * N_stubborn_A +
            [opinion_B] * N_stubborn_B
        )

        # Shuffle to randomize presentation
        env_names, env_opinions = synchronized_shuffle(all_names, all_opinions)

        # Create prompt
        prompt = create_opinion_prompt(env_names, env_opinions, opinion_A, opinion_B)

        # Get LLM response (apply chat template for proper behavior)
        formatted_prompts = apply_chat_template_to_prompts(llm, [prompt])
        outputs = llm.generate(formatted_prompts, sampling_params)
        response = outputs[0].outputs[0].text
        chosen = parse_opinion_response(response, opinion_A, opinion_B)

        # Update opinion if valid
        if chosen is not None:
            current_regular_opinions[idx] = chosen

    # Sampling phase (record magnetizations)
    magnetizations = []

    for _ in range(sampling_steps):
        # Randomly select a regular agent
        idx = random.randint(0, N_regular - 1)

        # Generate unique names for all agents
        all_names = generate_unique_random_strings(length=2, num_strings=N_total)

        # Construct full opinion list
        all_opinions = (
            current_regular_opinions[:idx] +
            current_regular_opinions[idx+1:] +
            [opinion_A] * N_stubborn_A +
            [opinion_B] * N_stubborn_B
        )

        # Shuffle
        env_names, env_opinions = synchronized_shuffle(all_names, all_opinions)

        # Create prompt
        prompt = create_opinion_prompt(env_names, env_opinions, opinion_A, opinion_B)

        # Get LLM response (apply chat template for proper behavior)
        formatted_prompts = apply_chat_template_to_prompts(llm, [prompt])
        outputs = llm.generate(formatted_prompts, sampling_params)
        response = outputs[0].outputs[0].text
        chosen = parse_opinion_response(response, opinion_A, opinion_B)

        # Update opinion if valid
        if chosen is not None:
            current_regular_opinions[idx] = chosen

        # Calculate magnetization of regular agents
        N_A = current_regular_opinions.count(opinion_A)
        N_B = current_regular_opinions.count(opinion_B)
        m = (N_A - N_B) / N_regular
        magnetizations.append(m)

    # Calculate statistics
    avg_m = np.mean(magnetizations)
    std_m = np.std(magnetizations)

    return avg_m, std_m, current_regular_opinions


def calculate_external_field(N_stubborn_A: int, N_stubborn_B: int, N_regular: int) -> float:
    """
    Convert stubborn agent counts to external field strength.

    Args:
        N_stubborn_A: Number of stubborn agents supporting opinion A
        N_stubborn_B: Number of stubborn agents supporting opinion B
        N_regular: Number of regular (changeable) agents

    Returns:
        External field strength h
    """
    opinion_direction = 1  # Positive field favors opinion A
    N_stubborn = N_stubborn_A + N_stubborn_B

    # Net effect: positive when more stubborn agents support A
    net_stubborn = N_stubborn_A - N_stubborn_B

    # Normalize by total network size
    h = opinion_direction * net_stubborn / (N_regular + N_stubborn)

    return h


def simulate_hysteresis_cycle_direct_control(
    llm: LLM,
    sampling_params: SamplingParams,
    opinion_A: str,
    opinion_B: str,
    N_regular: int = 20,
    max_stubborn: int = 10,
    initial_equilibration_steps: int = 50,
    step_equilibration_steps: int = 15,
    sampling_steps: int = 30,
    n_repeats: int = 3,
    model_id: str = None,
    save_results: bool = True
) -> Dict:
    """
    Simulate hysteresis cycles by varying stubborn agents (external field).

    Performs multiple independent forward-backward sweeps and averages results.
    State continuity is maintained within each sweep to observe hysteresis.

    Args:
        llm: vLLM LLM instance
        sampling_params: vLLM SamplingParams for generation
        opinion_A: First opinion
        opinion_B: Second opinion
        N_regular: Number of regular (changeable) agents
        max_stubborn: Maximum number of stubborn agents per side
        initial_equilibration_steps: Equilibration steps for first field point
        step_equilibration_steps: Equilibration steps for subsequent points
        sampling_steps: Number of steps to sample magnetization at each field point
        n_repeats: Number of independent hysteresis cycles to average
        model_id: Model identifier for saving results
        save_results: Whether to save results to file

    Returns:
        Dictionary containing sweep data and metadata
    """
    print(f"\n{'='*60}")
    print(f"Starting Hysteresis Cycle Simulation")
    print(f"{'='*60}")
    print(f"Opinions: '{opinion_A}' vs '{opinion_B}'")
    print(f"Regular agents: N={N_regular}")
    print(f"Max stubborn agents: {max_stubborn}")
    print(f"Number of cycles: {n_repeats}")
    print(f"Temperature: {sampling_params.temperature}")

    # Define field sweep (stubborn agent counts)
    forward_stubborn_counts = list(range(-max_stubborn, max_stubborn + 1))
    backward_stubborn_counts = list(range(max_stubborn, -max_stubborn - 1, -1))

    # Convert to field values
    forward_fields = [
        calculate_external_field(
            max(0, count), max(0, -count), N_regular
        ) for count in forward_stubborn_counts
    ]
    backward_fields = [
        calculate_external_field(
            max(0, count), max(0, -count), N_regular
        ) for count in backward_stubborn_counts
    ]

    print(f"\nForward sweep: stubborn counts from {forward_stubborn_counts[0]} to {forward_stubborn_counts[-1]}")
    print(f"Backward sweep: stubborn counts from {backward_stubborn_counts[0]} to {backward_stubborn_counts[-1]}")

    # Storage for all cycles
    all_forward_magnetizations = []
    all_backward_magnetizations = []

    # Perform multiple independent cycles
    for cycle in range(n_repeats):
        print(f"\n{'='*60}")
        print(f"Cycle {cycle + 1}/{n_repeats}")
        print(f"{'='*60}")

        forward_mags = []
        forward_stds = []

        # Initialize with random 50/50 split
        current_opinions = (
            [opinion_A] * (N_regular // 2) +
            [opinion_B] * (N_regular - N_regular // 2)
        )
        random.shuffle(current_opinions)

        print("\nForward sweep (increasing field)...")

        # Forward sweep
        for i, count in enumerate(forward_stubborn_counts):
            N_stubborn_A = max(0, count)
            N_stubborn_B = max(0, -count)

            # Use longer equilibration for first point, shorter for others
            eq_steps = initial_equilibration_steps if i == 0 else step_equilibration_steps

            avg_m, std_m, current_opinions = simulate_hysteresis_step_with_state(
                llm, sampling_params,
                current_opinions,
                N_stubborn_A, N_stubborn_B,
                opinion_A, opinion_B,
                eq_steps, sampling_steps
            )

            forward_mags.append(avg_m)
            forward_stds.append(std_m)

            print(f"  Field {i+1}/{len(forward_stubborn_counts)}: "
                  f"stubborn={count:+3d}, m={avg_m:+.3f}±{std_m:.3f}")

        all_forward_magnetizations.append(forward_mags)

        print("\nBackward sweep (decreasing field)...")

        # Backward sweep (continues from forward sweep endpoint)
        backward_mags = []
        backward_stds = []

        for i, count in enumerate(backward_stubborn_counts):
            N_stubborn_A = max(0, count)
            N_stubborn_B = max(0, -count)

            avg_m, std_m, current_opinions = simulate_hysteresis_step_with_state(
                llm, sampling_params,
                current_opinions,
                N_stubborn_A, N_stubborn_B,
                opinion_A, opinion_B,
                step_equilibration_steps, sampling_steps
            )

            backward_mags.append(avg_m)
            backward_stds.append(std_m)

            print(f"  Field {i+1}/{len(backward_stubborn_counts)}: "
                  f"stubborn={count:+3d}, m={avg_m:+.3f}±{std_m:.3f}")

        all_backward_magnetizations.append(backward_mags)

    # Calculate averages across cycles
    forward_magnetizations = np.mean(all_forward_magnetizations, axis=0)
    backward_magnetizations = np.mean(all_backward_magnetizations, axis=0)

    # Calculate standard errors (error of the mean across cycles)
    forward_errors = np.std(all_forward_magnetizations, axis=0) / np.sqrt(n_repeats)
    backward_errors = np.std(all_backward_magnetizations, axis=0) / np.sqrt(n_repeats)

    print(f"\n{'='*60}")
    print("Hysteresis cycle complete!")
    print(f"{'='*60}")

    # Prepare results dictionary
    sweep_data = {
        'forward_stubborn_counts': forward_stubborn_counts,
        'backward_stubborn_counts': backward_stubborn_counts,
        'forward_fields': forward_fields,
        'backward_fields': backward_fields,
        'forward_magnetizations': forward_magnetizations.tolist(),
        'backward_magnetizations': backward_magnetizations.tolist(),
        'forward_errors': forward_errors.tolist(),
        'backward_errors': backward_errors.tolist(),
        'all_forward_data': all_forward_magnetizations,
        'all_backward_data': all_backward_magnetizations,
        'N_regular': N_regular,
        'max_stubborn': max_stubborn,
        'temperature': sampling_params.temperature,
        'opinion_A': opinion_A,
        'opinion_B': opinion_B,
        'model_id': model_id,
        'initial_equilibration_steps': initial_equilibration_steps,
        'step_equilibration_steps': step_equilibration_steps,
        'sampling_steps': sampling_steps,
        'n_cycles': n_repeats
    }

    # Save results
    if save_results and model_id is not None:
        model_name = model_id.split("/")[-1]
        results_dir = Path(model_name) / "results_hysteresis_vllm" / f"N={N_regular}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save data
        filename = f"hysteresis_{opinion_A}_vs_{opinion_B}_T{sampling_params.temperature:.2f}.txt"
        filepath = results_dir / filename

        with open(filepath, 'w') as f:
            f.write("# Hysteresis Cycle Data\n")
            f.write(f"# Opinion A: {opinion_A}\n")
            f.write(f"# Opinion B: {opinion_B}\n")
            f.write(f"# N_regular: {N_regular}\n")
            f.write(f"# Temperature: {sampling_params.temperature}\n")
            f.write(f"# Cycles: {n_repeats}\n")
            f.write("stubborn_count,field,magnetization_forward,error_forward,magnetization_backward,error_backward\n")

            for i in range(len(forward_stubborn_counts)):
                f.write(f"{forward_stubborn_counts[i]},{forward_fields[i]:.6f},"
                       f"{forward_magnetizations[i]:.6f},{forward_errors[i]:.6f},"
                       f"{backward_magnetizations[-(i+1)]:.6f},{backward_errors[-(i+1)]:.6f}\n")

        print(f"Results saved to: {filepath}")

    return sweep_data


def plot_hysteresis_results(sweep_data: Dict, save_path: Optional[str] = None):
    """
    Plot hysteresis cycle results.

    Args:
        sweep_data: Dictionary containing sweep data from simulate_hysteresis_cycle_direct_control
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot vs stubborn count
    ax1.errorbar(
        sweep_data['forward_stubborn_counts'],
        sweep_data['forward_magnetizations'],
        yerr=sweep_data['forward_errors'],
        fmt='o-', color='blue', capsize=3, label='Forward sweep'
    )
    ax1.errorbar(
        sweep_data['backward_stubborn_counts'],
        sweep_data['backward_magnetizations'],
        yerr=sweep_data['backward_errors'],
        fmt='s-', color='red', capsize=3, label='Backward sweep'
    )
    ax1.set_xlabel('Stubborn Agent Count (positive = more A)', fontsize=12)
    ax1.set_ylabel('Magnetization m', fontsize=12)
    ax1.set_title(f"Hysteresis: {sweep_data['opinion_A']} vs {sweep_data['opinion_B']}", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    # Plot vs field
    ax2.errorbar(
        sweep_data['forward_fields'],
        sweep_data['forward_magnetizations'],
        yerr=sweep_data['forward_errors'],
        fmt='o-', color='blue', capsize=3, label='Forward sweep'
    )
    ax2.errorbar(
        sweep_data['backward_fields'],
        sweep_data['backward_magnetizations'],
        yerr=sweep_data['backward_errors'],
        fmt='s-', color='red', capsize=3, label='Backward sweep'
    )
    ax2.set_xlabel('External Field h', fontsize=12)
    ax2.set_ylabel('Magnetization m', fontsize=12)
    ax2.set_title(f"Hysteresis: {sweep_data['opinion_A']} vs {sweep_data['opinion_B']}", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    # Calculate hysteresis area (rough estimate using trapezoidal rule)
    forward_mags = np.array(sweep_data['forward_magnetizations'])
    backward_mags = np.array(sweep_data['backward_magnetizations'])[::-1]  # Reverse for alignment
    fields = np.array(sweep_data['forward_fields'])

    area_forward = np.trapz(forward_mags, fields)
    area_backward = np.trapz(backward_mags, fields)
    hysteresis_area = abs(area_forward - area_backward)

    fig.suptitle(f"N={sweep_data['N_regular']}, T={sweep_data['temperature']:.2f}, "
                f"Hysteresis area ≈ {hysteresis_area:.4f}", fontsize=10, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def run_hysteresis_experiment(
    model_id: str,
    opinion_pairs_file: str = "opinion_pairs.csv",
    N_regular: int = 50,
    max_stubborn: int = 30,
    initial_equilibration_steps: int = 50,
    step_equilibration_steps: int = 15,
    sampling_steps: int = 30,
    n_repeats: int = 1,
    temperature: float = 0.2,
    max_new_tokens: int = 16,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.95,
    max_model_len: Optional[int] = None,
    save_results: bool = True,
    generate_plots: bool = True,
    skip_existing: bool = True
) -> Dict[str, Dict]:
    """
    Run hysteresis experiments for multiple opinion pairs from a CSV file.

    Processes opinion pairs sequentially, with option to skip already completed pairs.
    Each hysteresis cycle varies the number of stubborn agents from -max_stubborn to
    +max_stubborn, measuring the magnetization response.

    Args:
        model_id: HuggingFace model identifier
        opinion_pairs_file: Path to comma-separated CSV file with header and Opinion_A, Opinion_B columns
        N_regular: Number of regular (changeable) agents
        max_stubborn: Maximum number of stubborn agents per side
        initial_equilibration_steps: Equilibration steps for first field point
        step_equilibration_steps: Equilibration steps for subsequent field points
        sampling_steps: Number of steps to sample magnetization at each field point
        n_repeats: Number of independent cycles to average
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate per response
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization (0.0-1.0)
        max_model_len: Maximum model sequence length (reduces KV cache if set)
        save_results: Whether to save results to files
        generate_plots: Whether to generate plots for each pair
        skip_existing: Whether to skip opinion pairs that already have results

    Returns:
        Dictionary mapping opinion pair strings to their sweep_data dictionaries
    """
    print(f"\n{'='*60}")
    print(f"Starting Hysteresis Experiment")
    print(f"{'='*60}")
    print(f"Model: {model_id}")
    print(f"Regular agents: N={N_regular}")
    print(f"Max stubborn agents: {max_stubborn}")
    print(f"Temperature: {temperature}")
    print(f"Cycles per pair: {n_repeats}")
    print(f"Skip existing: {skip_existing}")

    # Initialize vLLM model
    print("\nInitializing vLLM model...")
    llm_kwargs = {
        "model": model_id,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
        print(f"Setting max_model_len to {max_model_len}")

    llm = LLM(**llm_kwargs)

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_new_tokens
    )

    # Warmup
    print("Performing warmup...")
    warmup_prompts = ["Test prompt" for _ in range(3)]
    warmup_formatted = apply_chat_template_to_prompts(llm, warmup_prompts)
    _ = llm.generate(warmup_formatted, sampling_params)
    print("Warmup complete!")

    # Load opinion pairs
    opinion_pairs = pd.read_csv(opinion_pairs_file)
    print(f"\nLoaded {len(opinion_pairs)} opinion pairs from {opinion_pairs_file}")

    # Setup results directory
    model_name = model_id.split("/")[-1]
    results_dir = Path(model_name) / "results_hysteresis_vllm" / f"N={N_regular}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing results if skip_existing is True
    existing_pairs = set()
    if skip_existing:
        existing_files = list(results_dir.glob("hysteresis_*_T*.txt"))
        for f in existing_files:
            # Extract opinion pair from filename
            fname = f.stem  # e.g., "hysteresis_liberal_vs_conservative_T0.20"
            existing_pairs.add(fname)
        print(f"Found {len(existing_pairs)} existing result files")

    # Store all results
    all_results = {}

    # Process each opinion pair
    for idx, row in opinion_pairs.iterrows():
        opinion_A = row['Opinion_A']
        opinion_B = row['Opinion_B']

        # Check if already processed
        expected_filename = f"hysteresis_{opinion_A}_vs_{opinion_B}_T{temperature:.2f}"
        if skip_existing and expected_filename in existing_pairs:
            print(f"\n[{idx+1}/{len(opinion_pairs)}] Skipping '{opinion_A}' vs '{opinion_B}' (already exists)")
            continue

        print(f"\n{'='*60}")
        print(f"Opinion pair {idx+1}/{len(opinion_pairs)}")
        print(f"'{opinion_A}' vs '{opinion_B}'")
        print(f"{'='*60}")

        try:
            # Run hysteresis cycle
            sweep_data = simulate_hysteresis_cycle_direct_control(
                llm=llm,
                sampling_params=sampling_params,
                opinion_A=opinion_A,
                opinion_B=opinion_B,
                N_regular=N_regular,
                max_stubborn=max_stubborn,
                initial_equilibration_steps=initial_equilibration_steps,
                step_equilibration_steps=step_equilibration_steps,
                sampling_steps=sampling_steps,
                n_repeats=n_repeats,
                model_id=model_id,
                save_results=save_results
            )

            # Generate plot if requested
            if generate_plots:
                plot_path = results_dir / f"plot_{opinion_A}_vs_{opinion_B}.png"
                plot_hysteresis_results(sweep_data, save_path=str(plot_path))

            # Store result
            pair_key = f"{opinion_A}_vs_{opinion_B}"
            all_results[pair_key] = sweep_data

            print(f"\nCompleted: '{opinion_A}' vs '{opinion_B}'")

        except Exception as e:
            print(f"\nERROR processing '{opinion_A}' vs '{opinion_B}': {e}")
            continue

    # Print summary
    print(f"\n{'='*60}")
    print("Hysteresis Experiment Complete!")
    print(f"{'='*60}")
    print(f"Successfully processed: {len(all_results)} opinion pairs")
    print(f"Results saved to: {results_dir}")

    return all_results


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Configuration
    MODEL_ID = "Qwen/Qwen3-32B-Instruct"  # Change to your desired model
    TENSOR_PARALLEL_SIZE = 1  # Number of GPUs
    GPU_MEMORY_UTILIZATION = 0.9

    # Example 1: Single Agent Simulation
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Agent Simulation")
    print("="*60)

    llm_single = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True
    )

    sampling_params_single = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=16
    )

    magnetization_history = simulate_opinion_dynamics(
        llm=llm_single,
        sampling_params=sampling_params_single,
        Na_init=5,
        N=10,
        t_max=100,
        opinion1="pizza",
        opinion2="pasta",
        model_id=MODEL_ID,
        sim_id=0,
        save_results=True
    )

    # Example 2: Full Opinion Dynamics (requires opinion_pairs.csv)
    print("\n" + "="*60)
    print("EXAMPLE 2: Full Opinion Dynamics")
    print("="*60)

    # Uncomment to run if you have opinion_pairs.csv
    """
    results = run_experiment(
        model_id=MODEL_ID,
        opinion_pairs_file="opinion_pairs.csv",
        N=10,
        N_sim=100,
        batch_size=64,
        temperature=0.2,
        max_new_tokens=16,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        save_results=True,
        generate_plots=True
    )
    """

    # Example 3: Single Hysteresis Cycle
    print("\n" + "="*60)
    print("EXAMPLE 3: Single Hysteresis Cycle")
    print("="*60)

    llm_hyst = LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True
    )

    sampling_params_hyst = SamplingParams(
        temperature=0.3,
        top_p=0.95,
        max_tokens=16
    )

    sweep_data = simulate_hysteresis_cycle_direct_control(
        llm=llm_hyst,
        sampling_params=sampling_params_hyst,
        opinion_A="liberal",
        opinion_B="conservative",
        N_regular=20,
        max_stubborn=10,
        initial_equilibration_steps=50,
        step_equilibration_steps=15,
        sampling_steps=30,
        n_repeats=3,
        model_id=MODEL_ID,
        save_results=True
    )

    # Plot hysteresis results
    model_name = MODEL_ID.split("/")[-1]
    plot_path = Path(model_name) / "results_hysteresis_vllm" / f"N={sweep_data['N_regular']}" / \
                f"plot_{sweep_data['opinion_A']}_vs_{sweep_data['opinion_B']}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    plot_hysteresis_results(sweep_data, save_path=str(plot_path))

    # Example 4: Multiple Opinion Pairs Hysteresis (from CSV)
    print("\n" + "="*60)
    print("EXAMPLE 4: Multiple Opinion Pairs Hysteresis")
    print("="*60)

    # Uncomment to run if you have opinion_pairs.csv
    """
    all_hysteresis_results = run_hysteresis_experiment(
        model_id=MODEL_ID,
        opinion_pairs_file="opinion_pairs.csv",
        N_regular=50,
        max_stubborn=30,
        initial_equilibration_steps=50,
        step_equilibration_steps=15,
        sampling_steps=30,
        n_repeats=1,
        temperature=0.2,
        max_new_tokens=16,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        save_results=True,
        generate_plots=True,
        skip_existing=True  # Skip opinion pairs that already have results
    )
    """

    print("\n" + "="*60)
    print("All examples complete!")
    print("="*60)
