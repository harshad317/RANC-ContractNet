"""Build a NeurIPS-style supplementary artifact bundle."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence


BUNDLE_STEM = "ranc_contractnet_neurips2027_artifact"
DEFAULT_OUTPUT = Path("dist") / f"{BUNDLE_STEM}.zip"
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)

ROOT_FILES = [
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    ".zenodo.json",
]
SOURCE_ROOTS = ["src/ranc_contractnet", "experiments", "scripts", "tests", "paper/neurips2027"]
REQUIRED_PATHS = ROOT_FILES + SOURCE_ROOTS

EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "mlruns",
    "venv",
    "wandb",
}
EXCLUDED_FILE_PATTERNS = {
    ".DS_Store",
    "*.log",
    "*.pyc",
    "*.pyo",
    "ContractNet.pdf",
}
RAW_DATA_SUFFIXES = {
    ".arff",
    ".data",
    ".feather",
    ".joblib",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pq",
}
SUMMARY_OUTPUT_SUFFIXES = {".csv", ".json", ".md", ".tex"}
FORBIDDEN_CONTENT_TOKENS = {
    "/" + "Users" + "/",
    "hpu" + "4454",
}
@dataclass(frozen=True)
class ArtifactBuildResult:
    zip_path: Path
    sha256_path: Path
    file_count: int
    total_uncompressed_bytes: int
    zip_bytes: int


@dataclass(frozen=True)
class PackagedFile:
    source_path: Path
    archive_path: Path
    data: bytes


def _relative(path: Path, root: Path) -> Path:
    return path.resolve().relative_to(root.resolve())


def _uses_excluded_dir(relative_path: Path) -> bool:
    return any(part in EXCLUDED_DIR_NAMES for part in relative_path.parts)


def _matches_excluded_file(relative_path: Path) -> bool:
    name = relative_path.name
    return any(fnmatch.fnmatch(name, pattern) for pattern in EXCLUDED_FILE_PATTERNS)


def is_excluded_path(relative_path: Path) -> bool:
    if _uses_excluded_dir(relative_path):
        return True
    if _matches_excluded_file(relative_path):
        return True
    return relative_path.suffix.lower() in {".pyc", ".pyo"}


def is_summary_output_path(relative_path: Path) -> bool:
    parts = relative_path.parts
    if not parts or parts[0] != "outputs":
        return False
    if len(parts) >= 2 and (
        parts[1].startswith("artifact_dry_run") or parts[1].startswith("review_check")
    ):
        return False
    if len(parts) >= 2 and parts[1] == "smoke":
        return False
    if len(parts) >= 2 and parts[1] == "paper_render":
        if relative_path in {
            Path("outputs/paper_render/main.pdf"),
            Path("outputs/paper_render/paper_render_report.md"),
            Path("outputs/paper_render/paper_render_report.json"),
        }:
            return True
        if (
            len(parts) == 4
            and parts[:3] == ("outputs", "paper_render", "preview")
            and relative_path.suffix.lower() == ".png"
        ):
            return True
        return False
    if "runs" in parts:
        return False
    if len(parts) != 3:
        return False
    if relative_path.suffix.lower() in RAW_DATA_SUFFIXES:
        return False
    if relative_path.suffix.lower() not in SUMMARY_OUTPUT_SUFFIXES:
        return False
    if fnmatch.fnmatch(relative_path.name, "*_seed*.*"):
        return False
    return not is_excluded_path(relative_path)


def _iter_files(path: Path, root: Path) -> Iterable[Path]:
    if path.is_file():
        relative = _relative(path, root)
        if not path.is_symlink() and not is_excluded_path(relative):
            yield relative
        return
    if not path.exists():
        return
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        relative = _relative(candidate, root)
        if not is_excluded_path(relative):
            yield relative


def collect_artifact_files(
    root: Path,
    include_outputs: bool = True,
) -> List[Path]:
    root = root.resolve()
    files = set()
    for name in ROOT_FILES:
        path = root / name
        if path.exists():
            files.update(_iter_files(path, root))
    for name in SOURCE_ROOTS:
        path = root / name
        files.update(_iter_files(path, root))
    if include_outputs:
        outputs = root / "outputs"
        if outputs.exists():
            for candidate in sorted(outputs.rglob("*")):
                if not candidate.is_file() or candidate.is_symlink():
                    continue
                relative = _relative(candidate, root)
                if is_summary_output_path(relative):
                    files.add(relative)
    return sorted(files, key=lambda item: item.as_posix())


def validate_required_paths(root: Path) -> None:
    missing = [name for name in REQUIRED_PATHS if not (root / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Cannot build artifact; missing required paths: {joined}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _packaged_files(root: Path, relative_files: Sequence[Path]) -> List[PackagedFile]:
    packaged = [
        PackagedFile(
            source_path=relative,
            archive_path=relative,
            data=(root / relative).read_bytes(),
        )
        for relative in relative_files
    ]
    archive_paths = [item.archive_path.as_posix() for item in packaged]
    duplicates = sorted({path for path in archive_paths if archive_paths.count(path) > 1})
    if duplicates:
        joined = ", ".join(duplicates[:10])
        raise ValueError(f"Artifact archive path collision: {joined}")
    return sorted(packaged, key=lambda item: item.archive_path.as_posix())


def _scan_package_hygiene(packaged_files: Sequence[PackagedFile]) -> None:
    violations: List[str] = []
    tokens = set(FORBIDDEN_CONTENT_TOKENS)
    for packaged in packaged_files:
        path_text = packaged.archive_path.as_posix()
        for token in tokens:
            if token in path_text:
                violations.append(f"{path_text}: path contains {token!r}")
        try:
            text = packaged.data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for token in tokens:
            if token in text:
                violations.append(f"{path_text}: content contains {token!r}")
    if violations:
        detail = "\n".join(f"- {item}" for item in violations[:20])
        raise ValueError(f"Package hygiene sanity check failed:\n{detail}")


def _file_records(packaged_files: Sequence[PackagedFile]) -> List[dict]:
    records = []
    for packaged in packaged_files:
        records.append(
            {
                "path": packaged.archive_path.as_posix(),
                "size": len(packaged.data),
                "sha256": sha256_bytes(packaged.data),
            }
        )
    return records


def _render_manifest(records: Sequence[dict], *, include_outputs: bool) -> str:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total = sum(int(record["size"]) for record in records)
    lines = [
        "# RANC-ContractNet NeurIPS 2027 Artifact Manifest",
        "",
        f"- Generated UTC: {generated}",
        f"- Files: {len(records)}",
        f"- Uncompressed bytes: {total}",
        f"- Source roots: {', '.join(SOURCE_ROOTS)}",
        f"- Root files: {', '.join(ROOT_FILES)}",
        f"- Output summaries included: {str(include_outputs)}",
        "",
        "## Hygiene Policy",
        "",
        "- Excludes caches, bytecode, local OS files, build directories, and virtual environments.",
        "- Excludes raw OpenML run folders under `outputs/*/runs/`.",
        "- Excludes local smoke-test outputs under `outputs/smoke/`.",
        "- Excludes raw dataset-like binary/data suffixes such as `.pq`, `.parquet`, `.npy`, and `.pkl`.",
        "- Excludes generated per-seed outlier audit dumps matching `*_seed*.*`.",
        "- Includes `outputs/paper_render/main.pdf` only when the optional render tier has produced it.",
        "- Includes generated render preview PNGs under `outputs/paper_render/preview/` when present.",
        "- Excludes local validation reports under `outputs/review_check/` and `outputs/artifact_dry_run/`.",
        "- Runs an anonymization sanity check for local absolute-path/user tokens.",
        "",
        "## File Inventory",
        "",
        "| Path | Bytes | SHA256 |",
        "| --- | ---: | --- |",
    ]
    for record in records:
        lines.append(f"| `{record['path']}` | {record['size']} | `{record['sha256']}` |")
    return "\n".join(lines) + "\n"


def _render_reproduce() -> str:
    return """# Reproduce RANC-ContractNet Artifact

This bundle contains source code, tests, configs, paper scaffold files, and
generated summary artifacts. It intentionally does not contain private data,
OpenML raw dataset caches, virtual environments, or local build caches.

Start with `paper/neurips2027/artifact_eval.md` for tiered reviewer commands,
expected outputs, runtime estimates, and pass/fail criteria.

For the fast reviewer path, run:

```bash
python3 scripts/review_check.py --tier 1
```

For a clean extracted-bundle reviewer simulation after rebuilding the zip, run:

```bash
python3 scripts/review_check.py --tier dryrun
```

The dry-run report is written locally under `outputs/artifact_dry_run/` and is
not inserted back into the zip, because doing so would change the zip hash being
reported.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev,experiments]"
```

Install the optional neural dependencies only when running neural pilots:

```bash
python3 -m pip install -e ".[torch]"
```

## Fast Validation

```bash
python3 scripts/review_check.py --tier 1
python3 -m pytest
python3 -m compileall src experiments tests
```

## Regenerate Paper-Facing Synthetic Artifacts

```bash
python3 experiments/paper_results_runner.py
```

## Regenerate Network-Dependent Public Artifacts

```bash
python3 experiments/openml_runner.py --config experiments/configs/openml_public.yaml
```

OpenML datasets are fetched at runtime. The repository stores metrics, tables,
audit reports, task metadata, and exclusion logs only; raw public datasets are
not written into the artifact bundle.

## Rebuild This Bundle

```bash
python3 experiments/package_artifact.py
python3 scripts/review_check.py --tier dryrun
```
"""


def _render_sha256s(records: Sequence[dict], generated_files: Sequence[tuple[str, bytes]]) -> str:
    lines = [f"{record['sha256']}  {record['path']}" for record in records]
    for name, data in generated_files:
        lines.append(f"{sha256_bytes(data)}  {name}")
    return "\n".join(lines) + "\n"


def _write_zip_entry(handle: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    handle.writestr(info, data)


def build_artifact(
    root: Path,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    include_outputs: bool = True,
    max_size_bytes: int = 100 * 1024 * 1024,
) -> ArtifactBuildResult:
    root = root.resolve()
    output_path = output_path.resolve()
    validate_required_paths(root)
    relative_files = collect_artifact_files(root, include_outputs=include_outputs)
    packaged_files = _packaged_files(root, relative_files)
    _scan_package_hygiene(packaged_files)
    records = _file_records(packaged_files)
    total_uncompressed = sum(int(record["size"]) for record in records)
    if total_uncompressed > max_size_bytes:
        raise ValueError(
            f"Artifact payload is {total_uncompressed} bytes, exceeding limit {max_size_bytes} bytes."
        )

    manifest = _render_manifest(records, include_outputs=include_outputs).encode("utf-8")
    reproduce = _render_reproduce().encode("utf-8")
    generated = [
        ("ARTIFACT_MANIFEST.md", manifest),
        ("REPRODUCE.md", reproduce),
    ]
    sha256s = _render_sha256s(records, generated).encode("utf-8")
    generated.append(("SHA256SUMS", sha256s))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w") as archive:
        for packaged in packaged_files:
            arcname = f"{BUNDLE_STEM}/{packaged.archive_path.as_posix()}"
            _write_zip_entry(archive, arcname, packaged.data)
        for name, data in generated:
            _write_zip_entry(archive, f"{BUNDLE_STEM}/{name}", data)

    zip_hash = sha256_file(output_path)
    sha_path = output_path.with_suffix(".sha256")
    sha_path.write_text(f"{zip_hash}  {output_path.name}\n", encoding="utf-8")
    return ArtifactBuildResult(
        zip_path=output_path,
        sha256_path=sha_path,
        file_count=len(packaged_files) + len(generated),
        total_uncompressed_bytes=total_uncompressed,
        zip_bytes=output_path.stat().st_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-size-mb", type=float, default=100.0)
    parser.add_argument("--no-outputs", action="store_true", help="Exclude generated outputs summaries.")
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUT

    result = build_artifact(
        Path.cwd(),
        output,
        include_outputs=not args.no_outputs,
        max_size_bytes=int(args.max_size_mb * 1024 * 1024),
    )
    print(result.zip_path)
    print(result.sha256_path)
    print(f"files={result.file_count}")
    print(f"payload_bytes={result.total_uncompressed_bytes}")
    print(f"zip_bytes={result.zip_bytes}")


if __name__ == "__main__":
    main()
