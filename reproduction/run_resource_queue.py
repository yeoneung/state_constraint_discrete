"""Persistent CPU/GPU experiment queue with logs, utilization and failure records.

Run from a hidden background Python process on Windows. Child commands are
argument arrays, never shell strings. Work directories and all artifacts are
inside this submission. A failed run is recorded and never counted as complete.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def inside(path):
    path = Path(str(path).replace("\\", "/")).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Path outside submission: {path}")
    return path


def dump(path, payload):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def run(manifest_path, gpu_slots=2, cpu_slots=2):
    manifest_path = inside(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folder = inside(manifest_path.parent/"queue")
    folder.mkdir(exist_ok=True)
    pending, active, records = list(manifest["jobs"]), [], []
    limits = {"gpu": gpu_slots, "cpu": cpu_slots}
    start = time.monotonic()
    header = {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "limits": limits, "status": "running"}
    dump(folder/"status.json", {**header, "records": records})
    last_sample = -100.
    utilization = (folder/"utilization.csv").open("a", encoding="utf-8")
    utilization.write("elapsed_s,gpu_util_pct,memory_util_pct,memory_used_mib,power_w\n")
    try:
        while pending or active:
            external_gpu = 0
            for status_file in manifest.get("shared_gpu_status_files", []):
                status_path = inside(ROOT/status_file)
                if status_path.exists():
                    external = json.loads(status_path.read_text(encoding="utf-8"))
                    if external.get("status") == "running":
                        external_gpu += sum(i.get("resource") == "gpu" for i in external.get("active", []))
            for item in active[:]:
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                job = item["job"]
                artifact = inside(ROOT/job["completion_artifact"])
                records.append({"id": job["id"], "resource": job["resource"],
                                "exit_code": code, "artifact_exists": artifact.exists(),
                                "status": "complete" if code == 0 and artifact.exists() else "failed",
                                "seconds": time.monotonic()-item["start"]})
                active.remove(item)
            for job in pending[:]:
                resource = job["resource"]
                if resource not in limits:
                    raise ValueError("Job resource must be gpu or cpu")
                capacity = max(0, limits[resource]-external_gpu) if resource == "gpu" else limits[resource]
                if sum(i["job"]["resource"] == resource for i in active) >= capacity:
                    continue
                artifact = inside(ROOT/job["completion_artifact"])
                if artifact.exists():
                    records.append({"id": job["id"], "resource": resource,
                                    "status": "previously_complete", "artifact_exists": True})
                    pending.remove(job)
                    continue
                script = inside(ROOT/job["script"])
                log = (folder/f"{job['id']}.log").open("w", encoding="utf-8")
                env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                           PYTHONUNBUFFERED="1", PYTHONUTF8="1")
                command = [sys.executable, "-u", str(script),
                           *[str(arg).replace("\\", "/") for arg in job.get("args", [])]]
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                active.append({"job": job, "process": process, "log": log, "start": time.monotonic()})
                pending.remove(job)
                print(f"Started {job['id']} pid={process.pid} ({resource})", flush=True)
            elapsed = time.monotonic()-start
            if elapsed-last_sample >= 5:
                try:
                    sample = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
                                             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    if sample.returncode == 0:
                        utilization.write(f"{elapsed:.3f},"+sample.stdout.strip().replace(" ", "")+"\n")
                        utilization.flush()
                except (OSError, subprocess.TimeoutExpired):
                    pass
                last_sample = elapsed
            dump(folder/"status.json", {**header, "elapsed_seconds": elapsed, "pending": [j["id"] for j in pending],
                 "active": [{"id": i["job"]["id"], "pid": i["process"].pid, "resource": i["job"]["resource"]} for i in active],
                 "records": records})
            if active or pending:
                time.sleep(1)
    finally:
        utilization.close()
    header["status"] = "failed" if any(r["status"] == "failed" for r in records) else "complete"
    dump(folder/"status.json", {**header, "elapsed_seconds": time.monotonic()-start,
                               "pending": [], "active": [], "records": records})
    return 1 if header["status"] == "failed" else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--gpu-slots", type=int, default=2)
    parser.add_argument("--cpu-slots", type=int, default=2)
    args = parser.parse_args()
    if min(args.gpu_slots, args.cpu_slots) < 1:
        parser.error("Slot counts must be positive")
    sys.exit(run(args.manifest, args.gpu_slots, args.cpu_slots))
