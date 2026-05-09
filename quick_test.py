"""
Quick test / minimal example for BPINN wellbore stability analysis.

This script verifies the installation and demonstrates the core workflow
on a small-scale problem (~2 minutes on CPU). It covers:
  1. Physics engine: Monte Carlo sampling and analytical wellbore stress
     for the Mohr-Coulomb failure criterion
  2. BPINN model: Construction, training (50 epochs), Bayesian inference
  3. Uncertainty quantification: posterior predictive distribution from
     Bayesian weight sampling

Usage
-----
    python quick_test.py
"""

import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from BPINN_wellbore_stability import (
    PhysicalConstants,
    DistributionConfig,
    InputSampler,
    WellborePhysics,
    BayesianBPINN,
    DataScaler,
    OutputScaler,
    BPINNLoss,
    train_bpinn,
)
from torch.utils.data import DataLoader, TensorDataset


def run_quick_test():
    print("=" * 70)
    print("BPINN Wellbore Stability — Quick Test")
    print("=" * 70)

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # 1. Physics engine test
    # ------------------------------------------------------------------
    print("[1/4] Testing physics engine ...")
    phys_const = PhysicalConstants(
        depth=4500.0, stress_regime="NF", alpha_p=0.3
    )
    dist_config = DistributionConfig()
    sampler = InputSampler(dist_config, phys_const)
    physics = WellborePhysics(phys_const, dist_config)

    N_train, N_val = 200, 50
    X_train_8d, depths_tr, aux_tr = sampler.sample_inputs(
        N_train, regime="NF", alpha_p_mean=0.3, return_depth=True
    )
    Y_train = physics.compute_true_outputs(
        X_train_8d, aux_data=aux_tr, failure_model="mohr"
    )
    X_val_8d, depths_va, aux_va = sampler.sample_inputs(
        N_val, regime="NF", alpha_p_mean=0.3, return_depth=True
    )
    Y_val = physics.compute_true_outputs(
        X_val_8d, aux_data=aux_va, failure_model="mohr"
    )

    print(f"  Training samples : {N_train}")
    print(f"  Validation samples: {N_val}")
    print(f"  Input shape  : {X_train_8d.shape}")
    print(f"  Output shape : {Y_train.shape}")
    print(f"  Y_train stats: mean={Y_train.mean(axis=0).round(4)}, "
          f"std={Y_train.std(axis=0).round(4)}")
    print("  [PASS] Physics engine OK\n")

    # ------------------------------------------------------------------
    # 2. Data normalisation
    # ------------------------------------------------------------------
    print("[2/4] Preparing data ...")
    X_train_in = np.column_stack([depths_tr, X_train_8d])
    X_val_in = np.column_stack([depths_va, X_val_8d])

    scaler = DataScaler()
    scaler.fit(X_train_in)
    X_train_norm = scaler.transform(X_train_in)
    X_val_norm = scaler.transform(X_val_in)
    scaler.to_torch(device)

    train_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_train_norm),
            torch.FloatTensor(Y_train),
        ),
        batch_size=64, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.FloatTensor(X_val_norm),
            torch.FloatTensor(Y_val),
        ),
        batch_size=64, shuffle=False,
    )
    print("  [PASS] Data preparation OK\n")

    # ------------------------------------------------------------------
    # 3. BPINN training (small scale)
    # ------------------------------------------------------------------
    print("[3/4] Training BPINN (50 epochs) ...")
    model = BayesianBPINN(
        input_dim=9, output_dim=4,
        hidden_dims=[64, 64, 32],
        prior_sigma=1.0,
        phys_const=phys_const,
    ).to(device)

    output_scaler = OutputScaler().to(device)
    loss_fn = BPINNLoss(
        physics=physics,
        output_scaler=output_scaler,
        lambda_data=1.0, lambda_phys=0.1,
        lambda_kl=0.0001, lambda_frac=5.0,
        scaler=scaler, failure_model="mohr",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6,
    )

    t0 = time.perf_counter()
    history = train_bpinn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        device=device,
        verbose=False,
        mc_val_samples=3,
        early_stopping_patience=20,
        early_stopping_min_delta=1e-4,
    )
    elapsed = time.perf_counter() - t0

    final_train = history["train_loss"][-1]
    final_val = history["val_loss"][-1]
    print(f"  Training time  : {elapsed:.1f} s")
    print(f"  Final train loss: {final_train:.4f}")
    print(f"  Final val loss  : {final_val:.4f}")
    print("  [PASS] Training OK\n")

    # ------------------------------------------------------------------
    # 4. Bayesian inference (uncertainty quantification)
    # ------------------------------------------------------------------
    print("[4/4] Bayesian inference (20 MC weight samples) ...")
    model.eval()
    X_test_torch = torch.FloatTensor(X_val_norm).to(device)
    num_mc = 20
    preds = []
    with torch.no_grad():
        for _ in range(num_mc):
            y_pred = model(X_test_torch)
            preds.append(y_pred.cpu().numpy())

    preds = np.stack(preds, axis=0)  # (num_mc, N_val, 4)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    output_names = [
        "SF_collapse", "SF_fracture",
        "Rho_collapse (g/cm3)", "Rho_fracture (g/cm3)",
    ]
    print(f"  {'Output':<25s} {'Mean(std)':<18s} {'Avg uncertainty':<18s}")
    print("  " + "-" * 61)
    for i, name in enumerate(output_names):
        print(f"  {name:<25s} {mean_pred[:, i].mean():>8.4f}          "
              f"{std_pred[:, i].mean():>8.4f}")
    print("  [PASS] Bayesian inference OK\n")

    # ------------------------------------------------------------------
    print("=" * 70)
    print("All tests passed. The environment is correctly configured.")
    print("=" * 70)
    print("\nTo run the full analysis (2 failure criteria x 9 scenarios):")
    print("    python BPINN_wellbore_stability.py")
    print("\nFor a faster check (~5 min on CPU):")
    print('    set BPINN_QUICK=1 && python BPINN_wellbore_stability.py')
    return 0


if __name__ == "__main__":
    raise SystemExit(run_quick_test())
