"""Separate mesh, penalty, localization and inexact-policy errors in 1D."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from state_constrained.aligned.problem import RunConfig
from state_constrained.aligned.reference import solve_reference
from state_constrained.aligned.train import write_json

ROOT = Path(__file__).resolve().parents[2]


def run():
    torch.set_num_threads(2)
    output = ROOT/"reproduction/aligned/reference_studies"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ["boundary_decay_1d", "directional_exit_1d"]:
        base = RunConfig(name=name, h=.005, epsilon=.03, training_radius=10., device="cpu")
        cases = [("mesh", replace(base, h=h)) for h in [.03, .015, .0075, .00375, .001875]]
        cases += [("penalty", replace(base, h=.001, epsilon=eps)) for eps in [.1, .03, .01, .003, .001]]
        cases += [("coupled", replace(base, h=eps/2, epsilon=eps)) for eps in [.1, .03, .01, .003]]
        cases += [("localization", replace(base, improvement_radius=radius)) for radius in [2.02, 2.04, 2.08, 2.2, 2.5, 3., 4.]]
        cases += [("outer_box", replace(base, improvement_radius=2.1, training_radius=radius)) for radius in [2.12, 2.2, 2.5, 3., 4., 6., 10.]]
        for idx, (study, cfg) in enumerate(cases):
            solution = solve_reference(cfg)
            report = {"name": name, "study": study, "config": cfg.asdict(), **solution["report"]}
            write_json(output/f"{name}_{idx:02d}_{study}.json", report)
            np.savez_compressed(output/f"{name}_{idx:02d}_{study}.npz", **{k: v for k, v in solution.items() if k != "report"})
            records.append(report)
            print(name, study, cfg.h, cfg.epsilon, report["reference_rms"], flush=True)
        for mismatch in [.001, .01, .05]:
            solution = solve_reference(replace(base, h=.02, epsilon=.04), evaluation_error=mismatch)
            report = {"name": name, "study": "inexact_evaluation", "imposed_mismatch_bound": mismatch, **solution["report"]}
            write_json(output/f"{name}_inexact_{mismatch}.json", report)
            records.append(report)
    write_json(output/"summary.json", records)
    print(output, flush=True)


if __name__ == "__main__":
    run()
