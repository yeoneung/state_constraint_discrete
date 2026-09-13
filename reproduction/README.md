# Experimental reproduction

Run all commands from the repository root. The main README covers installation,
artifact retrieval, figure generation, verification, and retraining.

## Individual training runs

```bash
python code/scripts/run_aligned.py reproduction/aligned/final_exact/directional_exit_1d_pi_seed1/config.json --output reproduction/replays/example_1d
python code/scripts/aligned_paired_obstacle.py reproduction/aligned/paired_obstacle/raw_seed1/config.json --output reproduction/replays/example_raw --objective raw
python code/scripts/aligned_paired_obstacle.py reproduction/aligned/paired_obstacle/grid_seed1/config.json --output reproduction/replays/example_grid --objective grid
```

Use new output directories. The configuration records architecture, batch sizes,
mesh, penalty, boxes, optimization stages, precision, and seed. The two obstacle
objectives share the same configuration and sampling streams.

## Independent validation

```bash
python reproduction/validate_aligned_runs.py reproduction/replays/run_01
python reproduction/validate_paired_obstacle.py reproduction/replays/example_raw
python reproduction/validate_paired_obstacle.py reproduction/replays/example_grid
python reproduction/paired_feedback_values.py
```

The first command validates completed child run directories. It reports
float64 errors and performs the full finite-lattice policy checks for 1D PI.
The paired validator records frozen-policy values, true greedy gaps, and all
test trajectories using seed 20260911. `paired_feedback_values.py` evaluates
the deployed greedy policies in the six published `paired_obstacle/` run
directories; it is intended for regenerating those published diagnostics.
For a new paired campaign, update a separate copy of the run manifest to point
to the newly validated runs before regenerating aggregate plots.

## Reference calculations

These commands regenerate the corresponding reference directories:

```bash
python code/scripts/aligned_reference_studies.py
python code/scripts/aligned_confinement_diagnostic.py
python code/scripts/aligned_error_mechanism.py
python code/scripts/aligned_obstacle_reference.py --h 0.01 --epsilon 0.01 --viscosity-ratio 2 --penalty-scale 25 --radius 1.7 --improvement-radius 1.4 --boundary-factor 0.5 --output reproduction/aligned/final_obstacle/reference_half
```

The signed decomposition uses saved neural checkpoints, common midpoint
Dirichlet data, and nested reference meshes. Its remaining-bias term includes
penalty, localization, and unresolved fine-mesh effects. The 2% stopping
criterion measures successive mesh differences and is not a continuum error
certificate.

The saved-result verifier is for the published artifact set, whose hashes
are fixed. After independently regenerating artifacts, compare numerical
quantities as well as hashes: file serialization and elapsed times can change
without changing a mathematical result.
