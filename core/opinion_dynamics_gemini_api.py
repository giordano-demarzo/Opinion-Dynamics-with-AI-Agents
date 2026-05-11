import random
import numpy as np
import traceback
import math
import string
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import os
import time
import asyncio
from google import genai
from google.genai import types

# Initialize Gemini client with API key
API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=API_KEY)

def generate_unique_random_strings(length, num_strings):
    """Generate a list of unique random strings."""
    generated_strings = set()
    chars = string.ascii_letters + string.digits

    while len(generated_strings) < num_strings:
        random_str = ''.join(random.choice(chars) for _ in range(length))
        generated_strings.add(random_str)

    return list(generated_strings)

def synchronized_shuffle(a, b):
    """Shuffle two lists in the same order"""
    assert len(a) == len(b)
    indices = list(range(len(a)))
    random.shuffle(indices)
    a_shuffled = [a[i] for i in indices]
    b_shuffled = [b[i] for i in indices]
    return a_shuffled, b_shuffled

def create_chatbot_single_sync(prompt, max_tokens=64, max_retries=3):
    """Create a single LLM response using Gemini API (synchronous)."""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable thinking
                )
            )

            reply = response.text

            if reply:
                return reply.strip()
            else:
                return ""

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"  ⚠️  API error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
            else:
                print(f"  ❌ API error after {max_retries} attempts: {str(e)[:100]}")
                return ""

    return ""

async def create_chatbot_single(prompt, max_tokens=64, max_retries=3):
    """Create a single LLM response using Gemini API (async wrapper)."""
    return await asyncio.to_thread(create_chatbot_single_sync, prompt, max_tokens, max_retries)

async def process_prompts_parallel(prompts, max_tokens=64, batch_size=50):
    """Process multiple prompts in parallel using async API calls."""

    tasks = [create_chatbot_single(prompt, max_tokens=max_tokens) for prompt in prompts]

    results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(tasks) + batch_size - 1) // batch_size
        print(f"  🚀 Batch {batch_num}/{total_batches}: Processing {len(batch)} prompts in parallel...")

        start_time = time.time()
        batch_results = await asyncio.gather(*batch)
        elapsed = time.time() - start_time

        results.extend(batch_results)
        print(f"     ✓ Completed in {elapsed:.2f}s ({len(batch)/elapsed:.1f} queries/sec)")

    return results

async def run_experiment_async(max_tokens=64, N_sim=100, N=10, parallel_batch_size=50):
    """Run the main experiment with parallel async API calls."""

    print(f"\n{'='*70}")
    print(f"🚀 Starting Gemini gemini-2.5-flash Experiment (PARALLEL MODE)")
    print(f"{'='*70}")
    print(f"Population size (N): {N}")
    print(f"Simulations per condition (N_sim): {N_sim}")
    print(f"Parallel batch size: {parallel_batch_size}")
    print(f"Max tokens per response: {max_tokens}")
    print(f"{'='*70}\n")

    # Calculate list_N using the provided formula
    #list_N = ((N / 50) * np.array([50, 48, 45, 40, 30, 27, 23, 20, 10, 5, 2, 0])).tolist()
    list_N = ((N / 50) * np.array([48, 45, 40, 30, 27, 23, 20, 10, 5, 2])).tolist()
    list_N = [int(np.round(x)) for x in list_N]
    list_N = list(dict.fromkeys(list_N))  # Remove duplicates
    list_N = [x for x in list_N if x <= N]

    print(f"Initial conditions (m0 values): {len(list_N)} points")
    print(f"Total queries per opinion pair: {len(list_N) * N_sim}\n")

    # Load opinion pairs from CSV
    opinion_pairs = pd.read_csv('opinion_pairs.csv')
    print(f"Loaded {len(opinion_pairs)} opinion pairs from CSV\n")

    # Main simulation loop
    for idx, row in opinion_pairs.iterrows():
        print(f"\n{'='*70}")
        print(f"Opinion Pair {idx+1}/{len(opinion_pairs)}: '{row['Opinion_A']}' vs '{row['Opinion_B']}'")
        print(f"{'='*70}")

        opinion_names_initial = [row['Opinion_A'], row['Opinion_B']]

        folder_path = f'gemini-2.5-flash-lite/results_single_agent/N={N}'
        os.makedirs(folder_path, exist_ok=True)

        file_path = os.path.join(folder_path,
                                f'transition_prob_{N}_opinions_{opinion_names_initial[0].replace(" ", "_")}.txt')

        if os.path.exists(file_path):
            os.remove(file_path)

        list_p = []

        for i, Na in enumerate(list_N):
            Nb = N - Na
            m0 = (Na - Nb) / N

            print(f"\n[{i+1}/{len(list_N)}] Processing m0 = {m0:+.2f} (Na={Na}, Nb={Nb})")

            # Build all prompts for this initial condition
            all_prompts = []
            for sim in range(N_sim):
                list_names = generate_unique_random_strings(2, N)
                opinions = [opinion_names_initial[0]] * Na + [opinion_names_initial[1]] * Nb
                random.shuffle(opinions)
                temp_names = list_names[:]
                temp_traits = opinions[:]
                temp_traits = [": "+str(x)+"\n" for x in temp_traits]
                temp_names, temp_traits = synchronized_shuffle(temp_names, temp_traits)

                opinion1 = opinion_names_initial[0]
                opinion2 = opinion_names_initial[1]
                rand_opinion_names = []
                r = random.random()
                if r < 0.5:
                    rand_opinion_names.append(opinion1)
                    rand_opinion_names.append(opinion2)
                else:
                    rand_opinion_names.append(opinion2)
                    rand_opinion_names.append(opinion1)

                prompt = (
                    "Below you can see the list of all the other AI agents with the opinion they support.\n"
                    "You must reply with the opinion you want to support.\n" +
                    #f"The available opinions are {rand_opinion_names[0]} and {rand_opinion_names[1]}.\n" +
                    "The opinion must be reported between square brackets.\n" +
                    "".join([f"{temp_names[j]}{temp_traits[j]}" for j in range(len(temp_names))]) +
                    "Reply only with the opinion you want to support, between square brakets."
                )

                all_prompts.append(prompt)

            # Process all prompts in parallel
            start_time = time.time()
            all_responses = await process_prompts_parallel(all_prompts, max_tokens=max_tokens,
                                                          batch_size=parallel_batch_size)
            elapsed = time.time() - start_time

            # Count responses
            counts_opinion = [0, 0]
            invalid_count = 0
            debug_responses = []  # Store first few for debugging
            debug_prompts = []  # Store corresponding prompts
            for idx, response in enumerate(all_responses):
                if idx < 3:  # Save first 3 responses for debugging
                    debug_responses.append(response)
                    debug_prompts.append(all_prompts[idx])
                if "[" in response and "]" in response:
                    y = response.partition("[")[2].partition("]")[0].strip()
                    if opinion_names_initial[0] == y:
                        counts_opinion[0] += 1
                    elif opinion_names_initial[1] == y:
                        counts_opinion[1] += 1
                    else:
                        invalid_count += 1
                else:
                    invalid_count += 1

            # Debug: print sample prompts and responses
            if debug_responses:
                print(f"  🔍 Sample prompt & response:")
                for idx in range(min(1, len(debug_responses))):  # Just show first one
                    print(f"     PROMPT [{idx+1}]:")
                    print(f"     {debug_prompts[idx][:300]}...")
                    print(f"     RESPONSE [{idx+1}]:")
                    print(f"     {debug_responses[idx][:150]}")
                    print()

            p = counts_opinion[0] / N_sim if N_sim > 0 else 0
            se = math.sqrt((p * (1 - p)) / N_sim) if N_sim > 0 else 0
            list_p.append(p)

            print(f"  ⏱️  Total time: {elapsed:.2f}s ({N_sim/elapsed:.1f} queries/sec)")
            print(f"  📊 Results: p={p:.3f}±{se:.3f}")
            print(f"     {opinion_names_initial[0]}: {counts_opinion[0]}/{N_sim} ({counts_opinion[0]/N_sim*100:.1f}%)")
            print(f"     {opinion_names_initial[1]}: {counts_opinion[1]}/{N_sim} ({counts_opinion[1]/N_sim*100:.1f}%)")
            if invalid_count > 0:
                print(f"     Invalid/Empty: {invalid_count}/{N_sim} ({invalid_count/N_sim*100:.1f}%)")

            with open(file_path, 'a') as file:
                file.write(f"{m0},{counts_opinion[0]},{counts_opinion[1]},{p},{se}\n")

            # Sleep to avoid API rate limits
            print(f"  ⏸️  Sleeping for 1 seconds to avoid rate limits...")
            await asyncio.sleep(1)

        # Plotting results for each opinion pair
        try:
            df = pd.read_csv(file_path, header=None)
            df = df.applymap(lambda x: float(str(x).strip()))

            plt.figure(figsize=(8, 6))
            plt.plot(df[0], df[3], marker='o', markersize=8, linewidth=2)
            plt.xlabel('Initial Opinion Bias (m0)', fontsize=12)
            plt.ylabel('Transition Probability (p)', fontsize=12)
            plt.title(f'{opinion_names_initial[0]} vs {opinion_names_initial[1]}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = os.path.join(folder_path,
                                    f'plot_{opinion_names_initial[0].replace(" ", "_")}_vs_{opinion_names_initial[1].replace(" ", "_")}.png')
            plt.savefig(plot_path, dpi=150)
            plt.close()

            print(f"\n✅ Completed: '{opinion_names_initial[0]}' vs '{opinion_names_initial[1]}'")
            print(f"   📄 Data saved to: {file_path}")
            print(f"   📊 Plot saved to: {plot_path}")

        except Exception as e:
            print(f"\n❌ Error plotting results: {e}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")

def run_experiment(max_tokens=64, N_sim=100, N=10, parallel_batch_size=50):
    """Wrapper to run async experiment in sync context."""
    asyncio.run(run_experiment_async(max_tokens, N_sim, N, parallel_batch_size))


# =============================================================================
# HYSTERESIS CYCLE GENERATION
# =============================================================================

def parse_opinion_response(response, opinion_A, opinion_B):
    """Parse the LLM response to extract the chosen opinion."""
    if "[" in response and "]" in response:
        chosen_opinion = response.partition("[")[2].partition("]")[0].strip()
        if chosen_opinion == opinion_A:
            return opinion_A
        elif chosen_opinion == opinion_B:
            return opinion_B
    return None


def create_opinion_prompt(agent_names, agent_opinions, opinion_A, opinion_B):
    """Create a prompt showing agents and their opinions."""
    prompt = (
        "Below you can see the list of all the other AI agents with the opinion they support.\n"
        "You must reply with the opinion you want to support.\n"
        "The opinion must be reported between square brackets.\n"
    )

    for name, opinion in zip(agent_names, agent_opinions):
        prompt += f"{name}: {opinion}\n"

    prompt += "Reply only with the opinion you want to support, between square brackets."

    return prompt


async def simulate_hysteresis_step_with_state_async(
    current_regular_opinions,
    N_stubborn_A,
    N_stubborn_B,
    opinion_A,
    opinion_B,
    equilibration_steps,
    sampling_steps,
    max_tokens=16
):
    """
    Simulate one field point in the hysteresis cycle (async version for Gemini).

    This function modifies current_regular_opinions IN PLACE.
    """
    N_regular = len(current_regular_opinions)
    N_total = N_regular + N_stubborn_A + N_stubborn_B

    # Equilibration phase
    for _ in range(equilibration_steps):
        idx = random.randint(0, N_regular - 1)

        # Agent sees all OTHER agents (N_total - 1)
        all_names = generate_unique_random_strings(2, N_total - 1)

        all_opinions = (
            current_regular_opinions[:idx] +
            current_regular_opinions[idx+1:] +
            [opinion_A] * N_stubborn_A +
            [opinion_B] * N_stubborn_B
        )

        env_names, env_opinions = synchronized_shuffle(all_names, all_opinions)
        prompt = create_opinion_prompt(env_names, env_opinions, opinion_A, opinion_B)

        response = await create_chatbot_single(prompt, max_tokens=max_tokens)
        chosen = parse_opinion_response(response, opinion_A, opinion_B)

        if chosen is not None:
            current_regular_opinions[idx] = chosen

    # Sampling phase
    magnetizations = []

    for _ in range(sampling_steps):
        idx = random.randint(0, N_regular - 1)

        # Agent sees all OTHER agents (N_total - 1)
        all_names = generate_unique_random_strings(2, N_total - 1)

        all_opinions = (
            current_regular_opinions[:idx] +
            current_regular_opinions[idx+1:] +
            [opinion_A] * N_stubborn_A +
            [opinion_B] * N_stubborn_B
        )

        env_names, env_opinions = synchronized_shuffle(all_names, all_opinions)
        prompt = create_opinion_prompt(env_names, env_opinions, opinion_A, opinion_B)

        response = await create_chatbot_single(prompt, max_tokens=max_tokens)
        chosen = parse_opinion_response(response, opinion_A, opinion_B)

        if chosen is not None:
            current_regular_opinions[idx] = chosen

        N_A = current_regular_opinions.count(opinion_A)
        N_B = current_regular_opinions.count(opinion_B)
        m = (N_A - N_B) / N_regular
        magnetizations.append(m)

    avg_m = np.mean(magnetizations)
    std_m = np.std(magnetizations)

    return avg_m, std_m, current_regular_opinions


def calculate_external_field(N_stubborn_A, N_stubborn_B, N_regular):
    """Convert stubborn agent counts to external field strength."""
    opinion_direction = 1
    N_stubborn = N_stubborn_A + N_stubborn_B
    net_stubborn = N_stubborn_A - N_stubborn_B
    h = opinion_direction * net_stubborn / (N_regular + N_stubborn)
    return h


async def simulate_hysteresis_cycle_direct_control_async(
    opinion_A,
    opinion_B,
    N_regular=20,
    max_stubborn=10,
    initial_equilibration_steps=50,
    step_equilibration_steps=15,
    sampling_steps=30,
    n_repeats=3,
    max_tokens=16,
    model_id="gemini-2.5-flash-lite",
    save_results=True
):
    """
    Simulate hysteresis cycles by varying stubborn agents (async version for Gemini).
    """
    print(f"\n{'='*60}")
    print(f"Starting Hysteresis Cycle Simulation (Gemini)")
    print(f"{'='*60}")
    print(f"Opinions: '{opinion_A}' vs '{opinion_B}'")
    print(f"Regular agents: N={N_regular}")
    print(f"Max stubborn agents: {max_stubborn}")
    print(f"Number of cycles: {n_repeats}")

    forward_stubborn_counts = list(range(-max_stubborn, max_stubborn + 1))
    backward_stubborn_counts = list(range(max_stubborn, -max_stubborn - 1, -1))

    forward_fields = [
        calculate_external_field(max(0, count), max(0, -count), N_regular)
        for count in forward_stubborn_counts
    ]
    backward_fields = [
        calculate_external_field(max(0, count), max(0, -count), N_regular)
        for count in backward_stubborn_counts
    ]

    print(f"\nForward sweep: stubborn counts from {forward_stubborn_counts[0]} to {forward_stubborn_counts[-1]}")
    print(f"Backward sweep: stubborn counts from {backward_stubborn_counts[0]} to {backward_stubborn_counts[-1]}")

    all_forward_magnetizations = []
    all_backward_magnetizations = []

    for cycle in range(n_repeats):
        print(f"\n{'='*60}")
        print(f"Cycle {cycle + 1}/{n_repeats}")
        print(f"{'='*60}")

        forward_mags = []
        forward_stds = []

        current_opinions = (
            [opinion_A] * (N_regular // 2) +
            [opinion_B] * (N_regular - N_regular // 2)
        )
        random.shuffle(current_opinions)

        print("\nForward sweep (increasing field)...")

        for i, count in enumerate(forward_stubborn_counts):
            N_stubborn_A = max(0, count)
            N_stubborn_B = max(0, -count)

            eq_steps = initial_equilibration_steps if i == 0 else step_equilibration_steps

            avg_m, std_m, current_opinions = await simulate_hysteresis_step_with_state_async(
                current_opinions,
                N_stubborn_A, N_stubborn_B,
                opinion_A, opinion_B,
                eq_steps, sampling_steps,
                max_tokens=max_tokens
            )

            forward_mags.append(avg_m)
            forward_stds.append(std_m)

            print(f"  Field {i+1}/{len(forward_stubborn_counts)}: "
                  f"stubborn={count:+3d}, m={avg_m:+.3f}±{std_m:.3f}")

        all_forward_magnetizations.append(forward_mags)

        print("\nBackward sweep (decreasing field)...")

        backward_mags = []
        backward_stds = []

        for i, count in enumerate(backward_stubborn_counts):
            N_stubborn_A = max(0, count)
            N_stubborn_B = max(0, -count)

            avg_m, std_m, current_opinions = await simulate_hysteresis_step_with_state_async(
                current_opinions,
                N_stubborn_A, N_stubborn_B,
                opinion_A, opinion_B,
                step_equilibration_steps, sampling_steps,
                max_tokens=max_tokens
            )

            backward_mags.append(avg_m)
            backward_stds.append(std_m)

            print(f"  Field {i+1}/{len(backward_stubborn_counts)}: "
                  f"stubborn={count:+3d}, m={avg_m:+.3f}±{std_m:.3f}")

        all_backward_magnetizations.append(backward_mags)

    forward_magnetizations = np.mean(all_forward_magnetizations, axis=0)
    backward_magnetizations = np.mean(all_backward_magnetizations, axis=0)

    forward_errors = np.std(all_forward_magnetizations, axis=0) / np.sqrt(n_repeats)
    backward_errors = np.std(all_backward_magnetizations, axis=0) / np.sqrt(n_repeats)

    print(f"\n{'='*60}")
    print("Hysteresis cycle complete!")
    print(f"{'='*60}")

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
        'temperature': 0.0,
        'opinion_A': opinion_A,
        'opinion_B': opinion_B,
        'model_id': model_id,
        'initial_equilibration_steps': initial_equilibration_steps,
        'step_equilibration_steps': step_equilibration_steps,
        'sampling_steps': sampling_steps,
        'n_cycles': n_repeats
    }

    if save_results:
        from pathlib import Path
        results_dir = Path(model_id) / "results_hysteresis_gemini" / f"N={N_regular}"
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = f"hysteresis_{opinion_A}_vs_{opinion_B}.txt"
        filepath = results_dir / filename

        with open(filepath, 'w') as f:
            f.write("# Hysteresis Cycle Data\n")
            f.write(f"# Opinion A: {opinion_A}\n")
            f.write(f"# Opinion B: {opinion_B}\n")
            f.write(f"# N_regular: {N_regular}\n")
            f.write(f"# Cycles: {n_repeats}\n")
            f.write("stubborn_count,field,magnetization_forward,error_forward,magnetization_backward,error_backward\n")

            for i in range(len(forward_stubborn_counts)):
                f.write(f"{forward_stubborn_counts[i]},{forward_fields[i]:.6f},"
                       f"{forward_magnetizations[i]:.6f},{forward_errors[i]:.6f},"
                       f"{backward_magnetizations[-(i+1)]:.6f},{backward_errors[-(i+1)]:.6f}\n")

        print(f"Results saved to: {filepath}")

    return sweep_data


def plot_hysteresis_results(sweep_data, save_path=None):
    """Plot hysteresis cycle results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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

    forward_mags = np.array(sweep_data['forward_magnetizations'])
    backward_mags = np.array(sweep_data['backward_magnetizations'])[::-1]
    fields = np.array(sweep_data['forward_fields'])

    area_forward = np.trapz(forward_mags, fields)
    area_backward = np.trapz(backward_mags, fields)
    hysteresis_area = abs(area_forward - area_backward)

    fig.suptitle(f"N={sweep_data['N_regular']}, Hysteresis area ≈ {hysteresis_area:.4f}",
                fontsize=10, y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def simulate_hysteresis_cycle_direct_control(
    opinion_A,
    opinion_B,
    N_regular=20,
    max_stubborn=10,
    initial_equilibration_steps=50,
    step_equilibration_steps=15,
    sampling_steps=30,
    n_repeats=3,
    max_tokens=16,
    model_id="gemini-2.5-flash-lite",
    save_results=True
):
    """Wrapper to run async hysteresis experiment in sync context."""
    return asyncio.run(simulate_hysteresis_cycle_direct_control_async(
        opinion_A, opinion_B, N_regular, max_stubborn,
        initial_equilibration_steps, step_equilibration_steps,
        sampling_steps, n_repeats, max_tokens, model_id, save_results
    ))


if __name__ == "__main__":
    run_experiment(
        max_tokens=64,
        N_sim=100,
        N=50,
        parallel_batch_size=100
    )
