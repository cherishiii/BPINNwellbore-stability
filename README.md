# BPINN Wellbore Stability — Code for Reproducibility

> **Paper**: *Probabilistic Wellbore Stability Assessment for Underground Gas Storage in Depleted Carbonate Reservoirs with Fractured-Vuggy Heterogeneity: A Bayesian Physics-Informed Neural Network Approach*

## Overview

This repository contains the source code for a **Bayesian Physics-Informed Neural Network (BPINN)** framework that performs probabilistic wellbore stability analysis for underground gas storage (UGS) operations in depleted fractured-vuggy carbonate reservoirs.

The code implements:

- **Monte Carlo sampling** of geomechanical input parameters with physically consistent distributions.
- **Wellbore stress analysis** supporting two failure criteria used in the manuscript (Mohr–Coulomb and Mogi–Coulomb) and three stress regimes (Normal Fault, Strike-Slip, Reverse Fault).
- **Bayesian Neural Network** with variational inference (Bayes by Backprop) and physics-informed loss functions.
- **Uncertainty quantification** decomposing total variance into aleatoric (input parameter variability) and epistemic (model weight uncertainty) contributions.
- **Reliability–equivalent mud density curves** and safe mud-weight window estimation.

## File Description

| File | Description |
|------|-------------|
| `BPINN_wellbore_stability.py` | Core program containing all physics models, BPINN architecture, training, inference, and visualization (~4 600 lines). |
| `plot_style.py` | Shared Matplotlib style configuration (Times New Roman, journal-quality formatting). |
| `quick_test.py` | Minimal test script to verify installation and demonstrate the workflow (~2 min on CPU). |
| `requirements.txt` | Python package dependencies. |

## Requirements

- **Python** ≥ 3.8
- **PyTorch** ≥ 1.12 (CPU or CUDA)
- See `requirements.txt` for all dependencies.

## Installation

```bash
# 1. Create a virtual environment (recommended)
python -m venv bpinn_env

# Windows
bpinn_env\Scripts\activate
# Linux / macOS
source bpinn_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note on PyTorch**: If you have a CUDA-capable GPU, install the appropriate PyTorch build from https://pytorch.org/get-started/locally/ for significant speedup.

## Quick Test

Run the quick test to verify that your environment is correctly set up:

```bash
python quick_test.py
```

Expected output (approximately):

```
======================================================================
BPINN Wellbore Stability — Quick Test
======================================================================
Device: cpu

[1/4] Testing physics engine ...
  Training samples : 200
  Validation samples: 50
  ...
  [PASS] Physics engine OK

[2/4] Preparing data ...
  [PASS] Data preparation OK

[3/4] Training BPINN (50 epochs) ...
  Training time  : ~30 s
  ...
  [PASS] Training OK

[4/4] Bayesian inference (20 MC weight samples) ...
  ...
  [PASS] Bayesian inference OK

======================================================================
All tests passed. The environment is correctly configured.
======================================================================
```

If all four steps display `[PASS]`, the environment is ready.

## Full Reproduction

To reproduce the complete results presented in the paper (two failure criteria × nine stress scenarios):

```bash
python BPINN_wellbore_stability.py
```

This will:

1. Generate Monte Carlo training data for NF/SS/RF regimes with α_p ∈ {0.1, 0.3, 0.5}.
2. Train two BPINN models (one per failure criterion: Mohr–Coulomb and Mogi–Coulomb).
3. Compute reliability curves and compare BPINN predictions against analytical solutions.
4. Produce all figures and tables, saved under `results/` and `models/`.

**Estimated runtime**: ~30–60 min on GPU, ~2–4 hours on CPU.

For a faster check (~5 min on CPU) with reduced sample sizes and training epochs:

```bash
# Windows (cmd)
set BPINN_QUICK=1 && python BPINN_wellbore_stability.py

# Windows (PowerShell)
$env:BPINN_QUICK="1"; python BPINN_wellbore_stability.py

# Linux / macOS
BPINN_QUICK=1 python BPINN_wellbore_stability.py
```

## Output Structure

After a full run, the following files are produced:

```
models/
  bpinn_model_mohr.pth          # Trained model weights (Mohr–Coulomb)
  bpinn_model_mogi.pth          # Trained model weights (Mogi–Coulomb)
  scaler_mohr.pkl               # Input normalisation parameters
  scaler_mogi.pkl

results/
  input_distributions.png       # Input parameter probability distributions
  training_history_*.png        # Training loss curves per criterion
  reliability_comparison_*.png  # Physics vs BPINN reliability curves
  reliability_physics.png       # Analytical reliability curves
  rho_vs_inclination_*.png      # Mud density vs inclination angle
  criteria_comparison.png       # Comparative plot across the two criteria
  criterion_comparison.xlsx     # Quantitative comparison table
  computational_efficiency.xlsx # Benchmark timing results
  dataset_*.xlsx                # Training and validation datasets
```

## License

This code is provided for academic reproducibility purposes accompanying the manuscript.
