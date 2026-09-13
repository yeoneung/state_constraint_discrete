"""Independent double-precision diagnostics and exhaustive 1D policy checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code/src"))
sys.path.insert(0, str(ROOT/"code/scripts"))
from state_constrained.aligned.train import load_run, validate, write_json, _evaluate_chunks
from state_constrained.aligned.sampling import Sampler
from aligned_postprocess_1d import process as lattice_process


def process(run):
    run = Path(run)
    torch.set_num_threads(2)
    p, model, frozen, saved = load_run(run/"checkpoint_final.pt", dtype="float64")
    if (run/"evaluation.npz").exists():
        x = torch.from_numpy(np.load(run/"evaluation.npz")["x"]).double()
    else:
        x = Sampler(p, 918273, "cpu", torch.float64).domain(p.cfg.metric_points)
    box = Sampler(p, 3918273, "cpu", torch.float64).validation_box(p.cfg.validation_points)
    report = validate(p, model, frozen, x, box)
    report.update(validation_precision="float64", original_training_precision=saved["config"]["dtype"],
        domain_points=len(x), box_points=len(box), box_sampling_seed=3918273,
        checkpoint_sha256=hashlib.sha256((run/"checkpoint_final.pt").read_bytes()).hexdigest(),
        final_stage=saved["stage"], validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    prediction = _evaluate_chunks(model, x).numpy()
    exact = p.exact(x).numpy() if p.cfg.name != "benchmark_obstacle_2d" else np.empty(0)
    np.savez_compressed(run/"validation_float64.npz", x=x.numpy(), prediction=prediction, exact=exact,
                        box_points=box.numpy())
    write_json(run/"validation_float64.json", report)
    if p.dim == 1 and p.cfg.method == "pi":
        lattice_process(run)
    return {"run": run.name, **report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folders", nargs="*", type=Path,
                        default=[ROOT/"reproduction/aligned/final_exact", ROOT/"reproduction/aligned/final_cylinder"])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    runs = sorted({p.parent for folder in args.folders for p in folder.glob("*/checkpoint_final.pt")})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(process, runs))
    destination = ROOT/"results/validation/aligned_float64_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined = {r["run"]: r for r in json.loads(destination.read_text())} if destination.exists() else {}
    combined.update({r["run"]: r for r in results})
    write_json(destination, [combined[key] for key in sorted(combined)])
    print(f"Validated {len(results)} final networks; {destination}")
