"""Run the test suite with logs and all temporary artifacts inside submission."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    temporary = ROOT/"results/test_tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    log = ROOT/"results/validation/tests.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT/"code/src"), PYTHONUTF8="1",
               TMP=str(temporary), TEMP=str(temporary), TMPDIR=str(temporary))
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
            cwd=ROOT/"code", env=env, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    print("\n".join(log.read_text(encoding="utf-8").splitlines()[-5:]))
    sys.exit(result.returncode)
