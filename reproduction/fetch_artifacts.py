"""Download and verify the fixed experiment artifacts from the GitHub release."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded archive")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "reproduction/ARTIFACTS.json").read_text())
    expected = {entry["path"]: entry for entry in manifest["files"]}
    if not args.verify_only:
        path = args.archive
        if path is None:
            cache = ROOT / ".cache"
            cache.mkdir(exist_ok=True)
            path = cache / manifest["filename"]
            if not path.exists():
                temporary = path.with_suffix(".part")
                request = urllib.request.Request(manifest["url"], headers={"User-Agent": "state-constraint-reproduction"})
                with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
                if sha(temporary) != manifest["sha256"]:
                    raise ValueError("Downloaded archive SHA-256 mismatch")
                temporary.replace(path)
        if sha(path) != manifest["sha256"]:
            raise ValueError("Archive SHA-256 mismatch")
        with zipfile.ZipFile(path) as archive:
            if set(archive.namelist()) != set(expected) | {"ARTIFACT_MANIFEST.json"}:
                raise ValueError("Unexpected archive contents")
            if json.loads(archive.read("ARTIFACT_MANIFEST.json")) != manifest["files"]:
                raise ValueError("Archive manifest differs from the repository manifest")
            for name, entry in expected.items():
                target = (ROOT / name).resolve()
                if not target.is_relative_to(ROOT):
                    raise ValueError("Archive member escapes repository")
                if target.exists():
                    if sha(target) == entry["sha256"]:
                        continue
                    raise FileExistsError(f"An existing file differs from the release: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(target.suffix + ".part")
                with archive.open(name) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
                if sha(temporary) != entry["sha256"]:
                    raise ValueError(f"Extracted member hash mismatch: {name}")
                temporary.replace(target)
    for name, entry in expected.items():
        path = ROOT / name
        if not path.is_file() or sha(path) != entry["sha256"]:
            raise ValueError(f"Missing or modified artifact: {name}")
    sources = json.loads((ROOT / "reproduction/SOURCE_HASHES.json").read_text())
    for name, digest in sources.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"Scientific source differs from the experiment snapshot: {name}")
    print(f"Verified {len(expected)} artifacts and {len(sources)} scientific source files.")


if __name__ == "__main__":
    main()
