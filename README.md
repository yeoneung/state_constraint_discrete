# State-constrained discrete policy iteration

Reproducible experiments for **Penalty-Uniform Localization for
State-Constrained Policy Iteration**, by Yeongjong Kim, Jiwoong Jang, and Yeoneung Kim.

This repository contains the computational implementation, prescribed run
configurations, saved-result metadata, verification routines, and plotting code.
It covers 60 explicit-solution runs and six paired obstacle runs. Checkpoints and
numerical arrays are distributed in the
[experiment release](https://github.com/yeoneung/state_constraint_discrete/releases/tag/experiments-2026-09-13).
The manuscript and supplementary document are not included.

## Setup

Use Python 3.12. Install PyTorch for your CPU or CUDA device, then:

```bash
python -m pip install -r reproduction/requirements-aligned.txt
python -m pip install -e "./code[aligned]"
```

The recorded environment uses PyTorch 2.11.0+cu128, NumPy 2.5.2, SciPy 1.18.0,
Matplotlib 3.11.1, and PyYAML 6.0.3. The original hardware was an RTX 5070 GPU
and an i9-14900KF CPU. One-dimensional runs use CPU/float64; cylindrical and
obstacle runs use GPU/float32, with final diagnostics reevaluated in float64.
TF32 is disabled. Run the commands below from the repository root.

## Verify and regenerate the reported results

```bash
python reproduction/fetch_artifacts.py
python reproduction/run_aligned_tests.py
python reproduction/build_aligned_results.py
python reproduction/verify_aligned_package.py
```

The download is checked against `reproduction/ARTIFACTS.json`, including a
SHA-256 hash for every checkpoint and array. It contains all prescribed
intermediate stages, the final value/frozen-policy pairs, finite-grid reference
solutions, signed error fields, and complete obstacle paths, including failures.
The core scientific source is checked against `reproduction/SOURCE_HASHES.json`.

Plots and machine-readable results are generated in `figures/aligned/`.
LaTeX table fragments are generated in `results/tables/`; a LaTeX installation
is unnecessary. Numerical and test reports go to `results/validation/`.
The `reference_outputs/` directory contains the reported numerical summaries
for comparison. The verifier checks all 66 neural runs and 460 saved evaluation
stages; it does not require a manuscript, PDF, or journal template.

An already downloaded archive can be supplied with:

```bash
python reproduction/fetch_artifacts.py --archive /path/to/reproduction-artifacts-2026-09-13.zip
python reproduction/fetch_artifacts.py --verify-only
```

## Retrain the prescribed experiments

`reproduction/aligned/PAPER_RUNS.json` selects every reported neural run.
Each selected run directory contains the exact configuration, source snapshot,
precision, and saved metrics. The paired obstacle design is recorded in
`reproduction/aligned/paired_obstacle/manifest.json`.

```bash
python reproduction/replay_aligned_campaign.py --output reproduction/replays/run_01 --gpu-slots 2 --cpu-slots 2
```

The queue creates a new directory, runs all 66 prescribed neural configurations,
and records failures, timings, and hardware utilization. It keeps the published
artifacts unchanged. Retraining can vary across hardware and library versions;
equal nominal budgets do not imply equal optimizer evaluations or wall time.
Fresh training outputs need the validation steps described in
[reproduction/README.md](reproduction/README.md) before comparing final metrics.

## What the experiments establish

| Experiment | Code | Interpretation |
|---|---|---|
| Numerical leakage and localization | `code/scripts/aligned_confinement_diagnostic.py` | Two penalty sources on five meshes and all 15 pairs with h ≤ ε; finite-grid boundary and solver checks. |
| Reference sweeps | `code/scripts/aligned_reference_studies.py` | 62 configurations separating mesh, penalty, box, and iteration effects. |
| Signed error decomposition | `code/scripts/aligned_error_mechanism.py` | Evaluation, policy-iteration, mesh/viscosity, and remaining-bias fields on common nodes. Their RMS norms are not additive. |
| Explicit solutions | `code/scripts/run_aligned.py` | Five paired seeds for each benchmark and method, including cylindrical geometries in dimensions 2, 5, 10, and 20. |
| Paired obstacle evaluation | `code/scripts/aligned_paired_obstacle.py` | Raw frozen residual versus fitting computed frozen-policy grid values, with matched neural initialization, sample streams, precision, and nominal budgets. |

Both neural methods use the monotone centered operator, its matching greedy
control, bounded squared-distance penalties, separate training/improvement
boxes, and a shared inward exterior policy. Grid-assisted obstacle training
includes finite-grid Dirichlet information and additional sparse solves; its
targets are frozen-policy values, not analytic or optimal-value labels.

All seeds use the final prescribed stage. The obstacle tests share 200 starts,
three time steps, a goal radius of 0.15, and segment checks against every hole.
The raw controllers can stop near the goal while remaining outside that radius.
Finite trajectory tests do not prove universal continuous-time admissibility.
The cylindrical exact value depends on one aggregate coordinate and does not
demonstrate accuracy for a solution of high intrinsic dimension.
