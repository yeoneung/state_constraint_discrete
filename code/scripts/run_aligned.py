"""Run the revised shared operator: python scripts/run_aligned.py config.json."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from state_constrained.aligned.problem import RunConfig
from state_constrained.aligned.train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(RunConfig(**json.loads(args.config.read_text(encoding="utf-8"))), args.output)
