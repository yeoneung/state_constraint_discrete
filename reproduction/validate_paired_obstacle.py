"""Independent diagnostics for one completed paired obstacle run."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code/scripts"))
from validate_aligned_runs import process as validate_final
from validate_hybrid_policy_values import process as validate_policies
from aligned_obstacle_audit import audit
from state_constrained.aligned.train import write_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT/"reproduction/aligned/final_obstacle/reference_half")
    parser.add_argument("--sampling-seed", type=int, default=20260911)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    started = time.monotonic()
    while not (args.run/"metrics.json").exists():
        if time.monotonic()-started > 3600:
            raise TimeoutError(args.run)
        time.sleep(2)
    validate_final(args.run)
    validate_policies(args.run, args.reference)
    audit(args.run/"checkpoint_final.pt", args.run/"test_audit", heldout_count=200,
        sampling_seed=args.sampling_seed, split="test", threads=args.threads)
    write_json(args.run/"diagnostics_complete.json", {"status": "complete", "sampling_seed": args.sampling_seed})
