"""Verify files against upstream metadata and freeze reproducible download URLs."""
from pathlib import Path
import hashlib
import json
import urllib.request

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]


def git_blob(path):
    h = hashlib.sha1()
    h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    sources = json.loads((ROOT / "data/source-revisions.json").read_text())
    urls = {}
    hf_metadata = {}
    git_metadata = {}
    for directory in ("locomo-refined", "scriptmem"):
        info = sources[directory]
        repo = info["source"].removeprefix("https://github.com/").removesuffix(".git")
        request = urllib.request.Request(f'https://api.github.com/repos/{repo}/git/trees/{info["revision"]}?recursive=1', headers={"User-Agent": "text-memory-setup"})
        with urllib.request.urlopen(request) as response:
            tree = json.load(response)
        if tree.get("truncated"):
            raise ValueError(f"Incomplete upstream tree: {repo}")
        git_metadata[directory] = {f["path"]: f["sha"] for f in tree["tree"] if f["type"] == "blob"}
    for repo, info in sources.items():
        if "directory" not in info:
            continue
        meta = HfApi().dataset_info(repo, revision=info["revision"], files_metadata=True)
        hf_metadata[repo] = {f.rfilename: f for f in meta.siblings}
    by_directory = {v["directory"]: repo for repo, v in sources.items() if "directory" in v}
    by_directory["beam"] = "Mohammadta/BEAM"
    for path in sorted((ROOT / "data/public").rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        relative = path.relative_to(ROOT / "data/public")
        directory = relative.parts[0]
        filename = "/".join(relative.parts[1:])
        if directory in {"locomo-refined", "scriptmem"}:
            info = sources[directory]
            blob = git_metadata[directory][filename]
            if git_blob(path) != blob:
                raise ValueError(f"Git source mismatch: {relative}")
            repo = info["source"].removeprefix("https://github.com/").removesuffix(".git")
            url = f'https://raw.githubusercontent.com/{repo}/{info["revision"]}/{filename}'
        else:
            repo = by_directory[directory]
            if directory == "beam" and (filename.startswith("data/10M-") or filename == "10M-README.md"):
                repo = "Mohammadta/BEAM-10M"
                if filename == "10M-README.md":
                    filename = "README.md"
            file = hf_metadata[repo][filename]
            if file.lfs:
                with path.open("rb") as handle:
                    actual = hashlib.file_digest(handle, "sha256").hexdigest()
                expected = file.lfs.sha256
            else:
                actual, expected = git_blob(path), file.blob_id
            if actual != expected:
                raise ValueError(f"Hugging Face source mismatch: {relative}")
            url = f'https://huggingface.co/datasets/{repo}/resolve/{sources[repo]["revision"]}/{filename}'
        urls[str(path.relative_to(ROOT))] = url
    (ROOT / "data/download-urls.json").write_text(json.dumps(urls, indent=2, ensure_ascii=False) + "\n")
    print(f"Verified {len(urls)} files against pinned upstream metadata; download URLs recorded.")


if __name__ == "__main__":
    main()
