"""Replay every declared neural run into a fresh folder inside submission."""
import argparse
import json
from pathlib import Path
import sys
from run_resource_queue import run as queue

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-slots", type=int, default=2)
    parser.add_argument("--cpu-slots", type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("Replay output must be inside submission")
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT/"reproduction/aligned/PAPER_RUNS.json").read_text())
    selections = [(p, False) for p in manifest["exact_runs"]]
    jobs = []
    for relative, hybrid in selections:
        source = ROOT/relative
        config = json.loads((source/"config.json").read_text())
        name = source.name
        config_path = output/(name+".json")
        config_path.write_text(json.dumps(config, indent=2)+"\n")
        result = output/name
        jobs.append({"id": name, "resource": "gpu" if config["device"].startswith("cuda") else "cpu",
            "script": "code/scripts/aligned_preconditioned_obstacle.py" if hybrid else "code/scripts/run_aligned.py",
            "args": [config_path.relative_to(ROOT).as_posix(), "--output", result.relative_to(ROOT).as_posix()],
            "completion_artifact": (result/"metrics.json").relative_to(ROOT).as_posix()})
    for entry in manifest.get("paired_obstacle_runs", []):
        source = ROOT/entry["run"]
        name = "paired_"+source.name
        config_path = output/(name+".json")
        config_path.write_bytes((source/"config.json").read_bytes())
        result = output/name
        jobs.append({"id": name, "resource": "gpu",
            "script": "code/scripts/aligned_paired_obstacle.py",
            "args": [config_path.relative_to(ROOT).as_posix(), "--output", result.relative_to(ROOT).as_posix(),
                     "--objective", entry["objective"]],
            "completion_artifact": (result/"metrics.json").relative_to(ROOT).as_posix()})
    path = output/"manifest.json"
    path.write_text(json.dumps({"replay_of": "reproduction/aligned/PAPER_RUNS.json", "jobs": jobs}, indent=2)+"\n")
    sys.exit(queue(path, args.gpu_slots, args.cpu_slots))
