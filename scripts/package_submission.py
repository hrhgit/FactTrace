"""Package the formal source tree with per-file hashes; exclude local state and archives."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ('text_memory', 'tests', 'docs', 'deploy', 'examples', 'patches', 'vendor/aml')
# Keep the release package limited to the maintained runners. Historical audits and
# ablations stay outside the formal source tree even when they are useful locally.
FORMAL_SCRIPT_FILES = (
    'aml.py', 'build_mixed_eval.py', 'check_aml_endpoint.py', 'check_data.py',
    'package_submission.py', 'prepare_locomo.py', 'prepare_longmemeval.py',
    'prepare_personamem.py', 'record_sources.py', 'restore_data.py',
    'run_mixed_answers.py', 'run_mixed_memory.py', 'sample_eval_sets.py',
)
SOURCE_FILES = ('README.md', 'AGENTS.md', 'LICENSE', '.env.example', '.gitignore', '.python-version',
                'requirements.txt', 'requirements.lock.txt', 'pytest.ini',
                'data/source-revisions.json', 'data/official-source.json')


def source_files(root):
    paths = [root / name for name in SOURCE_FILES]
    paths += [root / 'scripts' / name for name in FORMAL_SCRIPT_FILES]
    paths += [p for name in SOURCE_DIRS for p in (root / name).rglob('*') if p.is_file()]
    return sorted({p for p in paths if p.is_file() and '__pycache__' not in p.parts
                   and not p.is_symlink() and (not p.name.startswith('.env') or p == root / '.env.example')})


def package(output, *, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    paths = source_files(root)
    # Only source assets are packaged; caches, databases and nested local outputs are never accepted.
    allowed_suffixes = {'.py', '.md', '.json', '.txt', '.ini', '.service', '.conf', '.patch', '.example', ''}
    paths = [p for p in paths if p.suffix in allowed_suffixes and
             not {'outputs', 'archive', 'state', '.venv', '.git', '.pytest_cache'} & set(p.relative_to(root).parts)]
    if output in paths:
        raise ValueError('Package output must not overwrite a source file')
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if not manifest or 'text_memory/__main__.py' not in manifest:
        raise ValueError('Formal source tree is missing')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in paths:
                info = zipfile.ZipInfo(str(path.relative_to(root)), date_time=(2026, 10, 30, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                bundle.writestr(info, path.read_bytes())
            info = zipfile.ZipInfo('MANIFEST.json', date_time=(2026, 10, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, json.dumps({'files_sha256': manifest}, indent=2) + '\n')
        with zipfile.ZipFile(temporary) as bundle:
            for name, expected in manifest.items():
                if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                    raise ValueError('Packaged file differs: ' + name)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {'output': str(output), 'files': len(manifest),
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(package(args.output), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
