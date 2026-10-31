"""Restore missing/modified public files using pinned upstream URLs and SHA-256."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    manifest = json.loads((ROOT / "data/files.sha256.json").read_text())
    urls = json.loads((ROOT / "data/download-urls.json").read_text())
    restored = 0
    for relative, expected in manifest.items():
        path = ROOT / relative
        if path.is_file() and path.stat().st_size == expected["bytes"] and digest(path) == expected["sha256"]:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        print("Downloading", relative, flush=True)
        with urllib.request.urlopen(urls[relative]) as response, temporary.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        if temporary.stat().st_size != expected["bytes"] or digest(temporary) != expected["sha256"]:
            raise ValueError(f"Upstream content mismatch: {relative}; original file preserved")
        os.replace(temporary, path)
        restored += 1
    source = json.loads((ROOT / "data/official-source.json").read_text())
    target = ROOT / "vendor/aml"
    if not target.exists():
        subprocess.run(["git", "clone", source["url"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", source["revision"]], check=True)
    revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if revision != source["revision"]:
        raise ValueError("Official pipeline revision differs from the pinned version; inspect before changing it")
    print(f"Restored {restored} public files; official pipeline revision verified.")


if __name__ == "__main__":
    main()
