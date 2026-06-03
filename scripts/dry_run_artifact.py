"""Run a clean extracted-bundle artifact dry run.

This script simulates a reviewer starting from the packaged supplementary zip:
it verifies the sidecar hash, extracts the bundle into a clean work directory,
checks the in-bundle SHA256SUMS file, runs the fast validation tier and optional
paper render tier from the extracted copy, then writes a local report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = Path("dist/ranc_contractnet_neurips2027_artifact.zip")
DEFAULT_OUTPUT_DIR = Path("outputs/artifact_dry_run")
BUNDLE_ROOT_NAME = "ranc_contractnet_neurips2027_artifact"


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: List[str]
    env_update: Dict[str, str] | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _redact(text: str, paths: Iterable[Path]) -> str:
    redacted = text
    for path in paths:
        try:
            redacted = redacted.replace(path.resolve().as_posix(), f"<{path.name or 'PATH'}>")
        except OSError:
            continue
    redacted = redacted.replace("/" + "Users" + "/", "/<USER_HOME>/")
    redacted = redacted.replace("hpu" + "4454", "<USER>")
    return redacted


def _relative_display(path: Path, root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return _redact(path.resolve().as_posix(), [root])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sidecar_hash(sidecar_path: Path) -> str | None:
    if not sidecar_path.exists():
        return None
    text = sidecar_path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return text.split()[0]


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        try:
            target.relative_to(destination)
        except ValueError as exc:
            raise ValueError(f"Unsafe zip member path: {member.filename}") from exc
    archive.extractall(destination)


def _find_bundle_root(extract_dir: Path) -> Path:
    preferred = extract_dir / BUNDLE_ROOT_NAME
    if preferred.is_dir():
        return preferred
    top_dirs = [path for path in extract_dir.iterdir() if path.is_dir()]
    if len(top_dirs) == 1:
        return top_dirs[0]
    raise FileNotFoundError(
        f"Expected one extracted bundle directory named {BUNDLE_ROOT_NAME!r}; found {len(top_dirs)}."
    )


def _run_command(
    spec: CommandSpec,
    *,
    cwd: Path,
    redact_paths: Sequence[Path],
    timeout_seconds: int | None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    env = os.environ.copy()
    if spec.env_update:
        env.update(spec.env_update)
    try:
        completed = subprocess.run(
            spec.argv,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        duration = time.perf_counter() - started
        display = " ".join(_redact(str(part), redact_paths) for part in spec.argv)
        return {
            "name": spec.name,
            "command": [_redact(str(part), redact_paths) for part in spec.argv],
            "command_display": display,
            "returncode": completed.returncode,
            "duration_seconds": round(duration, 3),
            "status": "passed" if completed.returncode == 0 else "failed",
            "stdout_tail": _tail(_redact(completed.stdout, redact_paths)),
            "stderr_tail": _tail(_redact(completed.stderr, redact_paths)),
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        display = " ".join(_redact(str(part), redact_paths) for part in spec.argv)
        return {
            "name": spec.name,
            "command": [_redact(str(part), redact_paths) for part in spec.argv],
            "command_display": display,
            "returncode": None,
            "duration_seconds": round(duration, 3),
            "status": "failed",
            "stdout_tail": _tail(_redact(stdout, redact_paths)),
            "stderr_tail": _tail(
                _redact(stderr + f"\nTimed out after {timeout_seconds} seconds.", redact_paths)
            ),
        }


def _artifact_checks(bundle_root: Path) -> List[Dict[str, Any]]:
    required = [
        Path("SHA256SUMS"),
        Path("pyproject.toml"),
        Path("outputs/paper_render/main.pdf"),
        Path("outputs/paper_render/paper_render_report.md"),
        Path("outputs/paper_render/paper_render_report.json"),
        Path("outputs/paper_render/preview/main.pdf.png"),
        Path("paper/neurips2027/STYLE_STATUS.md"),
    ]
    checks: List[Dict[str, Any]] = []
    for relative in required:
        path = bundle_root / relative
        checks.append(
            {
                "kind": "path",
                "path": relative.as_posix(),
                "status": "passed" if path.exists() else "failed",
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
            }
        )

    render_report = bundle_root / "outputs/paper_render/paper_render_report.md"
    if render_report.exists():
        text = render_report.read_text(encoding="utf-8", errors="replace")
        checks.extend(
            [
                {
                    "kind": "render_report",
                    "path": "outputs/paper_render/paper_render_report.md",
                    "status": "passed" if "Status: **passed**" in text else "failed",
                    "details": "contains passed status",
                },
                {
                    "kind": "render_report",
                    "path": "outputs/paper_render/paper_render_report.md",
                    "status": "passed" if "Style mode:" in text else "failed",
                    "details": "contains style mode",
                },
            ]
        )
    return checks


def _default_commands() -> List[CommandSpec]:
    tex_path = "/Library/TeX/texbin:" + os.environ.get("PATH", "")
    return [
        CommandSpec("sha256sums", ["shasum", "-a", "256", "-c", "SHA256SUMS"]),
        CommandSpec("pytest", [sys.executable, "-m", "pytest"]),
        CommandSpec("tier_1", [sys.executable, "scripts/review_check.py", "--tier", "1"]),
        CommandSpec(
            "render",
            [sys.executable, "scripts/review_check.py", "--tier", "render"],
            env_update={"PATH": tex_path},
        ),
    ]


def run_dry_run(
    *,
    zip_path: Path = DEFAULT_ZIP,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    root: Path = REPO_ROOT,
    timeout_seconds: int | None = 1800,
    checks_only: bool = False,
) -> Dict[str, Any]:
    root = root.resolve()
    zip_path = (root / zip_path).resolve() if not zip_path.is_absolute() else zip_path.resolve()
    output_dir = (root / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = _utc_now()
    work_dir = output_dir / "work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    extract_dir = work_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    zip_hash = sha256_file(zip_path) if zip_path.exists() else None
    sidecar_path = zip_path.with_suffix(".sha256")
    sidecar_hash = _read_sidecar_hash(sidecar_path)
    sidecar_check = {
        "kind": "sidecar_sha256",
        "path": _relative_display(sidecar_path, root),
        "status": "passed" if sidecar_hash is not None and sidecar_hash == zip_hash else "failed",
        "expected": sidecar_hash,
        "actual": zip_hash,
    }

    extraction_check: Dict[str, Any]
    bundle_root: Path | None = None
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path) as archive:
                _safe_extract(archive, extract_dir)
            bundle_root = _find_bundle_root(extract_dir)
            extraction_check = {
                "kind": "extract",
                "path": _relative_display(zip_path, root),
                "status": "passed",
                "bundle_root": BUNDLE_ROOT_NAME,
            }
        except Exception as exc:  # noqa: BLE001 - report must preserve extraction failures.
            extraction_check = {
                "kind": "extract",
                "path": _relative_display(zip_path, root),
                "status": "failed",
                "details": str(exc),
            }
    else:
        extraction_check = {
            "kind": "extract",
            "path": _relative_display(zip_path, root),
            "status": "failed",
            "details": "zip file is missing",
        }

    checks: List[Dict[str, Any]] = [sidecar_check, extraction_check]
    commands: List[Dict[str, Any]] = []
    if bundle_root is not None:
        checks.extend(_artifact_checks(bundle_root))
        if not checks_only:
            redact_paths = [root, output_dir, work_dir, bundle_root]
            for spec in _default_commands():
                commands.append(
                    _run_command(
                        spec,
                        cwd=bundle_root,
                        redact_paths=redact_paths,
                        timeout_seconds=timeout_seconds,
                    )
                )

    passed = all(check["status"] == "passed" for check in checks) and all(
        command["status"] == "passed" for command in commands
    )
    report = {
        "started_utc": started,
        "finished_utc": _utc_now(),
        "overall_status": "passed" if passed else "failed",
        "zip_path": _relative_display(zip_path, root),
        "zip_sha256": zip_hash,
        "sidecar_path": _relative_display(sidecar_path, root),
        "sidecar_sha256": sidecar_hash,
        "checks_only": checks_only,
        "extracted_bundle": BUNDLE_ROOT_NAME if bundle_root is not None else None,
        "commands": commands,
        "artifact_checks": checks,
        "report_policy": (
            "This local dry-run report is generated after packaging and is intentionally not "
            "inserted into the artifact zip, because doing so would change the zip hash being "
            "reported."
        ),
    }
    write_reports(report, output_dir)
    return report


def write_reports(report: Dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    json_path = output_dir / "extracted_bundle_report.json"
    md_path = output_dir / "extracted_bundle_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def render_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# Extracted Bundle Artifact Dry Run",
        "",
        f"- Overall status: **{report['overall_status']}**",
        f"- Started UTC: {report['started_utc']}",
        f"- Finished UTC: {report['finished_utc']}",
        f"- Zip path: `{report['zip_path']}`",
        f"- Zip SHA256: `{report['zip_sha256']}`",
        f"- Sidecar SHA256: `{report['sidecar_sha256']}`",
        f"- Checks-only mode: `{report['checks_only']}`",
        f"- Report policy: {report['report_policy']}",
        "",
        "## Artifact Checks",
        "",
        "| Kind | Path | Status | Details |",
        "| --- | --- | --- | --- |",
    ]
    for check in report["artifact_checks"]:
        detail_parts = []
        if "exists" in check:
            detail_parts.append(f"exists={check['exists']}")
        if "size_bytes" in check:
            detail_parts.append(f"bytes={check['size_bytes']}")
        if "details" in check:
            detail_parts.append(str(check["details"]))
        if check.get("kind") == "sidecar_sha256":
            detail_parts.append("sidecar matches zip hash")
        lines.append(
            f"| `{check['kind']}` | `{check.get('path', '')}` | {check['status']} | "
            f"{'; '.join(detail_parts)} |"
        )

    lines.extend(["", "## Commands", ""])
    if report["commands"]:
        lines.extend(
            [
                "| Name | Status | Seconds | Return code | Command |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for command in report["commands"]:
            returncode = "" if command["returncode"] is None else str(command["returncode"])
            lines.append(
                f"| {command['name']} | {command['status']} | "
                f"{command['duration_seconds']:.3f} | {returncode} | "
                f"`{command['command_display']}` |"
            )
    else:
        lines.append("No commands were run; checks-only mode was enabled.")

    if report["commands"]:
        lines.extend(["", "## Command Output Tails", ""])
        for command in report["commands"]:
            lines.append(f"### {command['name']}")
            lines.append("")
            lines.append("stdout tail:")
            lines.append("")
            lines.append("```text")
            lines.append(command.get("stdout_tail", "").rstrip())
            lines.append("```")
            lines.append("")
            lines.append("stderr tail:")
            lines.append("")
            lines.append("```text")
            lines.append(command.get("stderr_tail", "").rstrip())
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--checks-only",
        action="store_true",
        help="Only extract and verify packaged render artifacts; skip reviewer commands.",
    )
    args = parser.parse_args()

    report = run_dry_run(
        zip_path=args.zip_path,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        checks_only=args.checks_only,
    )
    print(args.output_dir / "extracted_bundle_report.md")
    print(args.output_dir / "extracted_bundle_report.json")
    print(f"status={report['overall_status']}")
    raise SystemExit(0 if report["overall_status"] == "passed" else 1)


if __name__ == "__main__":
    main()
