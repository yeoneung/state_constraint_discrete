"""Verify the declared revised artifacts; historical files are not baselines."""
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code/src"))
from state_constrained.aligned.problem import Problem, RunConfig
from state_constrained.aligned.operators import stencil_audit


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check():
    torch.set_num_threads(2)
    manifest = read(ROOT/"reproduction/aligned/PAPER_RUNS.json")
    selections = list(manifest["exact_runs"])
    selections += [entry["run"] for entry in manifest["paired_obstacle_runs"]]
    assert len(manifest["exact_runs"]) == 60 and len(selections) == 66
    rows = []
    for relative in selections:
        run = ROOT/relative
        cfg = read(run/"config.json")
        audit = stencil_audit(Problem(RunConfig(**cfg)))
        saved_audit = read(run/"operator_audit.json")
        assert abs(audit["minimum_transition_coefficient"]-saved_audit["minimum_transition_coefficient"]) < 1e-10
        history = read(run/"history.json")
        checkpoints = sorted(run.glob("checkpoint_[0-9][0-9].pt"))
        assert len(checkpoints) == len(history) == cfg["policy_iterations"]
        previous = None
        for stage, path in enumerate(checkpoints):
            saved = torch.load(path, map_location="cpu", weights_only=False)
            assert saved["stage"] == stage and saved["config"] == cfg
            frozen = saved["frozen_policy_model"]
            assert (frozen is None) == (stage == 0)
            if previous is not None:
                assert all(torch.equal(frozen[k], v) for k, v in previous.items())
            previous = saved["model"]
            assert all(torch.isfinite(t).all() for t in previous.values())
        final = torch.load(run/"checkpoint_final.pt", map_location="cpu", weights_only=False)
        assert final["stage"] == cfg["policy_iterations"]-1
        assert all(torch.equal(final["model"][k], v) for k, v in previous.items())
        for file, expected in final["source_sha256"].items():
            assert sha(run/"source"/file) == expected
            assert sha(ROOT/"code/src/state_constrained/aligned"/file) == expected
        if (run/"evaluator.json").exists():
            assert sha(run/"source/aligned_preconditioned_obstacle.py") == sha(ROOT/"code/scripts/aligned_preconditioned_obstacle.py")
            if read(run/"evaluator.json")["kind"].startswith("paired_"):
                assert sha(run/"source/aligned_paired_obstacle.py") == sha(ROOT/"code/scripts/aligned_paired_obstacle.py")
        validation = read(run/"validation_float64.json")
        assert validation["checkpoint_sha256"] == sha(run/"checkpoint_final.pt")
        assert validation["domain_points"] == cfg["metric_points"]
        if cfg["name"] != "benchmark_obstacle_2d":
            arrays = np.load(run/"validation_float64.npz")
            rms = np.sqrt(np.mean((arrays["prediction"]-arrays["exact"])**2))
            assert abs(rms-validation["rms"]) < 1e-12
        if cfg["dim"] == 1 and cfg["method"] == "pi":
            lattice = read(run/"lattice_validation.json")
            assert min(r["api_slack"] for r in lattice["stages"]) >= -1e-7
        rows.append({"run": relative, "stages": len(checkpoints), "checkpoint_sha256": sha(run/"checkpoint_final.pt")})
    paired = check_paired(manifest)
    mechanism = check_mechanism()
    confinement = check_confinement()
    figure_provenance = read(ROOT/"figures/aligned/provenance.json")
    assert figure_provenance["paper_runs"] == manifest
    assert figure_provenance["generator_sha256"] == sha(ROOT/"reproduction/build_aligned_results.py")
    assert figure_provenance["focused_generator_sha256"] == sha(ROOT/"reproduction/build_focused_results.py")
    assert figure_provenance["confinement_diagnostic_sha256"] == sha(ROOT/"reproduction/aligned/confinement_diagnostic/summary.json")
    for relative, expected in figure_provenance["artifacts"].items():
        assert sha(ROOT/relative) == expected
    result = {"status": "pass", "neural_runs": len(rows), "saved_evaluations": sum(r["stages"] for r in rows),
              "obstacle_controllers": 7, "runs": rows,
              "paired_obstacle": paired, "error_mechanism": mechanism, "confinement_diagnostic": confinement,
              "scope": "artifact consistency and numerical checks; not a mathematical proof of continuum accuracy"}
    path = ROOT/"results/validation/aligned_package_verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"runs", "obstacle"}}))
    return result


def check_paired(manifest):
    frozen = read(ROOT/manifest["paired_obstacle_manifest"])
    assert frozen["script_sha256"] == sha(ROOT/"code/scripts/aligned_paired_obstacle.py")
    folder = ROOT/"reproduction/aligned/paired_obstacle"
    pairs = []
    common_starts = None
    for seed in frozen["seeds"]:
        raw, grid = [folder/f"{arm}_seed{seed}" for arm in ["raw", "grid"]]
        assert read(raw/"config.json") == read(grid/"config.json")
        assert read(raw/"evaluator.json")["initial_model_sha256"] == read(grid/"evaluator.json")["initial_model_sha256"]
        for a, b in zip(read(raw/"history.json"), read(grid/"history.json")):
            assert a["sampled_batch_sha256"] == b["sampled_batch_sha256"]
            assert a["sampler_end_sha256"] == b["sampler_end_sha256"]
        pairs.append({"seed": seed, "config_and_initialization_match": True, "sample_stream_match": True})
    reference = np.load(ROOT/"reproduction/aligned/final_obstacle/reference_half/reference.npz")
    for name in [f"{arm}_seed{seed}" for seed in frozen["seeds"] for arm in ["raw", "grid"]]+["reference"]:
        run = folder/name
        report = read(run/"test_audit/obstacle_audit.json")
        assert report["split"] == "test" and report["heldout_sampling_seed"] == frozen["test_sampling_seed"]
        assert report["used_in_training"] is False and report["safety_filter"] is False
        assert [r["dt"] for r in report["results"]] == frozen["time_steps"]
        for result in report["results"]:
            arrays = np.load(run/f"test_audit/rollout_dt_{result['dt']}.npz")
            starts = arrays["starts"]
            if common_starts is None:
                common_starts = starts
            assert np.array_equal(common_starts, starts)
            assert arrays["paths"].shape == (round(20/result["dt"])+1, 205, 2)
            assert np.isfinite(arrays["paths"]).all()
            records = result["records"][5:]
            assert len(records) == 200
            for field, count in [("goal_reached", "goal_reached"), ("strict_segment_violation", "strict_segment_violations")]:
                assert sum(r[field] for r in records) == result["groups"]["heldout"][count]
            final_distance = np.linalg.norm(arrays["paths"][-1, 5:]-np.array([.95, .55]), axis=1)
            assert np.array_equal(final_distance <= .15, np.array([r["goal_reached"] for r in records]))
        if name != "reference":
            assert read(run/"hybrid_policy_validation.json")["minimum_api_slack"] >= -1e-7
            arrays = np.load(run/"final_feedback_value.npz")
            xx = np.stack(np.meshgrid(arrays["axis"], arrays["axis"], indexing="ij"), -1)
            p = Problem(RunConfig(**read(run/"config.json")))
            mask = p.in_domain(torch.from_numpy(xx.reshape(-1, 2))).numpy().ravel()
            error = abs((arrays["lower"]+arrays["upper"])/2-reference["value"])
            assert abs(error.ravel()[mask].max()-read(run/"final_feedback_value.json")["policy_value_error_domain_grid"]) < 1e-12
    return {"pairs": pairs, "test_seed": frozen["test_sampling_seed"], "common_starts": 200,
            "time_steps": frozen["time_steps"], "controller_count": 7}


def check_mechanism():
    folder = ROOT/"reproduction/aligned/error_mechanism"
    report = read(folder/"summary.json")
    assert report["source_sha256"] == sha(ROOT/"code/scripts/aligned_error_mechanism.py")
    assert len(report["rows"]) == 80
    maximum_closure = 0.
    for row in report["rows"]:
        arrays = np.load(folder/f"{row['name']}_seed{row['seed']}.npz")
        stage = row["stage"]
        terms = [arrays[f"evaluation_{stage:02d}"], arrays[f"iteration_{stage:02d}"], arrays["mesh"], arrays["bias_proxy"]]
        total = arrays[f"total_{stage:02d}"]
        closure = float(abs(total-sum(terms)).max())
        maximum_closure = max(maximum_closure, closure)
        assert closure < 1e-12
        for field, values in zip(["evaluation_rms", "iteration_rms", "mesh_rms", "bias_proxy_rms", "total_rms"], terms+[total]):
            assert abs(np.sqrt(np.mean(values**2))-row[field]) < 1e-12
        assert row["fine_successive_difference_rms"] <= .02*row["bias_proxy_rms"]
        checkpoint = ROOT/f"reproduction/aligned/final_exact/{row['name']}_pi_seed{row['seed']}/checkpoint_{stage:02d}.pt"
        assert sha(checkpoint) == row["checkpoint_sha256"]
        assert row["api_slack"] >= -1e-7
    return {"saved_evaluations": len(report["rows"]), "maximum_signed_identity_error": maximum_closure,
            "fine_reference_successive_change_below_2pct": True}


def check_confinement():
    folder = ROOT/"reproduction/aligned/confinement_diagnostic"
    report = read(folder/"summary.json")
    for relative,expected in report["source_sha256"].items():
        assert sha(ROOT/relative) == expected
    for name,expected in report["array_sha256"].items():
        assert sha(folder/name) == expected
    scales=report["design"]["scales"]
    assert {(r["h"],r["epsilon"]) for r in report["localization"]} == {(h,e) for h in scales for e in scales if h<=e}
    assert len(report["localization"]) == 15
    for row in report["localization"]:
        arrays=np.load(folder/row["array_file"])
        x=arrays["x"];target=arrays["target"]
        assert np.array_equal(target,np.abs(x)<=2+1e-12)
        h=row["h"];eps=row["epsilon"]
        assert abs(row["L"]-(2+.06+.02*np.log(.04/h)))<1e-14
        mids={k:(arrays[k+"_lower"]+arrays[k+"_upper"])/2 for k in ("localized","whole")}
        difference=float(np.max((mids["localized"]-mids["whole"])[target]))
        assert abs(difference-row["localization_error"])<1e-13
        assert abs(difference/h-row["localization_over_h"])<1e-12
        assert row["diagnostic_uncertainty"] < .001*difference
        for label in ("localized","whole"):
            for side in ("lower","upper"):
                v=arrays[label+"_"+side]
                grad=(v[2:]-v[:-2])/(2*h)
                action=np.sign(1-grad)
                if label=="localized":
                    action=np.where(np.abs(x[1:-1])<=row["L"]-h,action,-np.tanh(x[1:-1]))
                cost=1-action+np.minimum(np.maximum(np.abs(x[1:-1])-2,0)**2,.09)/eps
                residual=v[1:-1]-action*grad-row["N"]*(v[2:]+v[:-2]-2*v[1:-1])/h-cost
                assert float(np.max(np.abs(residual))) < 2e-8
                boundary=0. if side=="lower" else 2+.09/eps
                assert v[0]==v[-1]==boundary
    for row in report["leakage"]:
        arrays=np.load(folder/row["array_file"]);target=np.abs(arrays["x"])<=2+1e-12
        for name in ("distance","squared"):
            mean=(arrays[name+"_lower"]+arrays[name+"_upper"])/2
            assert abs(float(mean[target].max())-row[name+"_max"])<1e-14
            assert abs(row[name+"_max"]/row["h"]-row[name+"_over_h"])<1e-14
    return {"localization_pairs":15,"leakage_meshes":len(report["leakage"]),
            "max_localization_over_h":max(r["localization_over_h"] for r in report["localization"]),
            "max_relative_diagnostic_uncertainty":max(r["diagnostic_uncertainty"]/r["localization_error"] for r in report["localization"])}


if __name__ == "__main__":
    check()
