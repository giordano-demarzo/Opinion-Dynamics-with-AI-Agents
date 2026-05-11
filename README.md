# LLMs Opinion Dynamics — Replication Code and Data

Code and data to replicate the results of the paper  
**"Conformity Generates Collective Misalignment in AI Agents Societies"**.

---

## Repository structure

```
.
├── opinion_pairs.csv              # 100 opinion pairs used in experiments
├── requirements.txt               # Python dependencies
│
├── core/                          # Shared simulation engines
│   ├── opinion_dynamics_vllm.py       # vLLM backend (open-weight models)
│   ├── opinion_dynamics_gemini_api.py # Google Gemini backend
│   └── opinion_dynamics_openai_api.py # OpenAI backend
│
├── experiments/                   # Scripts that produce raw data
│   ├── run_transition_prob_experiment.py  # P(m) transition probability sweeps
│   ├── run_hysteresis_experiment.py       # Hysteresis simulations
│   └── run_tipping_point_experiment.py    # Tipping-point experiments
│
├── figures/                       # Scripts that read data and produce figures
│   ├── plot1.py             # Figure 1 — example opinion dynamics traces
│   ├── plot2.py             # Figure 2 — P(m) fits and data collapse
│   ├── plot3.py             # Figure 3 — phase diagram (all models)
│   ├── plot4.py             # Figure 4 — tipping point & hysteresis
│   ├── plot_tipping_point.py  # Figure 5 — theory vs observed z_c
│   └── plot_SI.py           # Supplementary Information figures
│
└── data/                          # Pre-computed simulation outputs
    ├── analysis_results_v2/
    │   └── all_fits_combined.csv      # Pre-fitted (β, h) for all models and pairs
    ├── results_robustness/            # Robustness checks (temperature, system size, prompt)
    └── <model_name>/                  # One directory per model
        ├── results_batched_vllm/N=50/    # Transition probability P(m) data (.txt per opinion)
        └── results_hysteresis_*/N=50/    # Hysteresis sweep data (.txt per pair)
            (results_hysteresis_vllm      for open-weight models)
            (results_hysteresis_gemini    for gemini-2.5-flash-lite)
            (results_hysteresis_openai    for gpt-5-mini)
```

---

## Models

| Model | Type | Directory |
|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | vLLM | `data/Llama-3.1-8B-Instruct/` |
| Qwen/Qwen2.5-14B-Instruct | vLLM | `data/Qwen2.5-14B-Instruct/` |
| Qwen/Qwen2.5-32B-Instruct | vLLM | `data/Qwen2.5-32B-Instruct/` |
| Qwen/Qwen3-14B | vLLM | `data/Qwen3-14B/` |
| Qwen/Qwen3-32B | vLLM | `data/Qwen3-32B/` |
| google/gemma-3-12b-it | vLLM | `data/gemma-3-12b-it/` |
| google/gemma-3-27b-it | vLLM | `data/gemma-3-27b-it/` |
| gemini-2.5-flash-lite | Gemini API | `data/gemini-2.5-flash-lite/` |
| gpt-5-mini | OpenAI API | `data/gpt-5-mini/` |

---

## Installation

```bash
pip install -r requirements.txt
```

**GPU note:** vLLM 0.19.1 requires a **CUDA 12.x** driver. CUDA 13.x is
incompatible. The `transformers` version must be ≥ 4.56.0 (earlier versions
lack the `ALLOWED_LAYER_TYPES` symbol that vLLM 0.19.1 imports).

**API keys:** Set environment variables before running API-backed models:
```bash
export GEMINI_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
```

---

## Reproducing the figures

All figure scripts read from `data/` and write outputs back to `data/`
(main figures) or `SI_figures/` (supplementary). Run from the repository root:

```bash
python figures/plot1.py            # Figure 1
python figures/plot2.py            # Figure 2
python figures/plot3.py            # Figure 3
python figures/plot4.py            # Figure 4
python figures/plot_tipping_point.py
python figures/plot_SI.py          # Supplementary → SI_figures/
```

---

## Reproducing the simulations

Experiments write results into `data/<model_name>/`. They require either a
GPU (vLLM models) or API access (Gemini, OpenAI).

### 1 — Transition probability P(m)

```bash
# Open-weight model (vLLM)
python experiments/run_transition_prob_experiment.py \
    --model google/gemma-3-27b-it \
    --opinion-pairs-file opinion_pairs.csv

# API model
python experiments/run_transition_prob_experiment.py \
    --model gemini-2.5-flash-lite \
    --opinion-pairs-file opinion_pairs.csv
```

Output: `data/<model_name>/results_batched_vllm/N=50/transition_prob_*.txt`

### 2 — Hysteresis simulations

```bash
python experiments/run_hysteresis_experiment.py \
    --model google/gemma-3-27b-it \
    --opinion-pairs-file opinion_pairs.csv \
    --max-stubborn 35
```

Output: `data/<model_name>/results_hysteresis_vllm/N=50/hysteresis_*_T0.20.txt`

### 3 — Tipping-point experiment

```bash
python experiments/run_tipping_point_experiment.py \
    --model google/gemma-3-27b-it \
    --opinion-A "gender self-identification" \
    --opinion-B "biological sex classification"
```
