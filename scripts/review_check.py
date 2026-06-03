"""Run reviewer-facing artifact checks and write consolidated reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("outputs/review_check")
DEFAULT_ZIP = Path("dist/ranc_contractnet_neurips2027_artifact.zip")


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: List[str]


@dataclass(frozen=True)
class TierSpec:
    name: str
    description: str
    commands: List[CommandSpec]
    expected_paths: List[Path]
    run_zip_hygiene: bool = False
    zip_path: Path = DEFAULT_ZIP


TIER_1 = TierSpec(
    name="1",
    description="Fast CPU validation",
    commands=[
        CommandSpec("pytest", [sys.executable, "-m", "pytest"]),
        CommandSpec("compileall", [sys.executable, "-m", "compileall", "src", "experiments", "tests"]),
    ],
    expected_paths=[],
)

TIER_2 = TierSpec(
    name="2",
    description="Synthetic paper artifact regeneration",
    commands=[CommandSpec("paper_results", [sys.executable, "experiments/paper_results_runner.py"])],
    expected_paths=[
        Path("outputs/outlier_pair/contract_causal_table.tex"),
        Path("outputs/outlier_pair/contract_delta_table.tex"),
        Path("outputs/outlier_pair/contract_statistical_paragraph.md"),
        Path("outputs/sparse/sparse_table.tex"),
        Path("outputs/temporal_drift/temporal_drift_table.tex"),
        Path("outputs/ablations/ablation_table.tex"),
        Path("outputs/benchmark/benchmark_table.tex"),
    ],
)

TIER_3 = TierSpec(
    name="3",
    description="Network-dependent OpenML/UCI regeneration",
    commands=[
        CommandSpec(
            "openml_public",
            [sys.executable, "experiments/openml_runner.py", "--config", "experiments/configs/openml_public.yaml"],
        )
    ],
    expected_paths=[
        Path("outputs/openml/openml_summary.csv"),
        Path("outputs/openml/openml_aggregate.csv"),
        Path("outputs/openml/openml_table.tex"),
        Path("outputs/openml/openml_task_metadata.csv"),
        Path("outputs/openml/exclusion_log.md"),
        Path("outputs/openml/openml_pairwise_deltas.csv"),
        Path("outputs/openml/openml_win_loss_table.tex"),
        Path("outputs/openml/openml_stats_paragraph.md"),
    ],
)

TIER_RENDER = TierSpec(
    name="render",
    description="Optional paper PDF render check",
    commands=[CommandSpec("paper_render", [sys.executable, "scripts/render_paper.py"])],
    expected_paths=[
        Path("outputs/paper_render/paper_render_report.md"),
        Path("outputs/paper_render/paper_render_report.json"),
    ],
)

TIER_RENDER_ANONYMOUS = TierSpec(
    name="render_anonymous",
    description="Optional anonymous paper PDF render check",
    commands=[
        CommandSpec(
            "paper_render_anonymous",
            [sys.executable, "scripts/render_paper.py", "--variant", "anonymous"],
        )
    ],
    expected_paths=[
        Path("outputs/paper_render/main_anonymous.pdf"),
        Path("outputs/paper_render/paper_render_report_anonymous.md"),
        Path("outputs/paper_render/paper_render_report_anonymous.json"),
    ],
)

TIER_DRYRUN = TierSpec(
    name="dryrun",
    description="Clean extracted-bundle reviewer dry run",
    commands=[CommandSpec("artifact_dry_run", [sys.executable, "scripts/dry_run_artifact.py"])],
    expected_paths=[
        Path("outputs/artifact_dry_run/extracted_bundle_report.md"),
        Path("outputs/artifact_dry_run/extracted_bundle_report.json"),
    ],
)

TIER_DRYRUN_ANONYMOUS = TierSpec(
    name="dryrun_anonymous",
    description="Clean extracted-bundle anonymous reviewer dry run",
    commands=[
        CommandSpec(
            "artifact_dry_run_anonymous",
            [
                sys.executable,
                "scripts/dry_run_artifact.py",
                "--zip-path",
                "dist/ranc_contractnet_neurips2027_artifact_anonymous.zip",
                "--output-dir",
                "outputs/artifact_dry_run_anonymous",
            ],
        )
    ],
    expected_paths=[
        Path("outputs/artifact_dry_run_anonymous/extracted_bundle_report.md"),
        Path("outputs/artifact_dry_run_anonymous/extracted_bundle_report.json"),
    ],
)

TIER_PACKAGE = TierSpec(
    name="package",
    description="Supplementary bundle packaging and zip hygiene",
    commands=[CommandSpec("package_artifact", [sys.executable, "experiments/package_artifact.py"])],
    expected_paths=[
        Path("dist/ranc_contractnet_neurips2027_artifact.zip"),
        Path("dist/ranc_contractnet_neurips2027_artifact.sha256"),
    ],
    run_zip_hygiene=True,
)

TIER_PACKAGE_ANONYMOUS = TierSpec(
    name="package_anonymous",
    description="Anonymous supplementary bundle packaging and zip hygiene",
    commands=[
        CommandSpec(
            "package_artifact_anonymous",
            [sys.executable, "experiments/package_artifact.py", "--identity-mode", "anonymous"],
        )
    ],
    expected_paths=[
        Path("dist/ranc_contractnet_neurips2027_artifact_anonymous.zip"),
        Path("dist/ranc_contractnet_neurips2027_artifact_anonymous.sha256"),
    ],
    run_zip_hygiene=True,
    zip_path=Path("dist/ranc_contractnet_neurips2027_artifact_anonymous.zip"),
)

TIER_BY_NAME = {
    "1": TIER_1,
    "2": TIER_2,
    "3": TIER_3,
    "render": TIER_RENDER,
    "render_anonymous": TIER_RENDER_ANONYMOUS,
    "dryrun": TIER_DRYRUN,
    "dryrun_anonymous": TIER_DRYRUN_ANONYMOUS,
    "package": TIER_PACKAGE,
    "package_anonymous": TIER_PACKAGE_ANONYMOUS,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _redact(text: str, root: Path) -> str:
    redacted = text.replace(root.as_posix(), "<REPO_ROOT>")
    redacted = redacted.replace("/" + "Users" + "/", "/<USER_HOME>/")
    redacted = redacted.replace("hpu" + "4454", "<USER>")
    return redacted


def command_to_display(argv: Sequence[str]) -> str:
    return " ".join(argv)


def run_command(spec: CommandSpec, cwd: Path, timeout_seconds: int | None = None) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            spec.argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        duration = time.perf_counter() - start
        redacted_argv = [_redact(str(part), cwd) for part in spec.argv]
        return {
            "name": spec.name,
            "command": redacted_argv,
            "command_display": command_to_display(redacted_argv),
            "returncode": completed.returncode,
            "duration_seconds": round(duration, 3),
            "status": "passed" if completed.returncode == 0 else "failed",
            "stdout_tail": _tail(_redact(completed.stdout, cwd)),
            "stderr_tail": _tail(_redact(completed.stderr, cwd)),
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        redacted_argv = [_redact(str(part), cwd) for part in spec.argv]
        return {
            "name": spec.name,
            "command": redacted_argv,
            "command_display": command_to_display(redacted_argv),
            "returncode": None,
            "duration_seconds": round(duration, 3),
            "status": "failed",
            "stdout_tail": _tail(_redact(stdout, cwd)),
            "stderr_tail": _tail(_redact(stderr + f"\nTimed out after {timeout_seconds} seconds.", cwd)),
        }


def check_expected_paths(root: Path, expected_paths: Iterable[Path]) -> List[Dict[str, Any]]:
    checks = []
    for relative in expected_paths:
        path = root / relative
        exists = path.exists()
        checks.append(
            {
                "kind": "path",
                "path": relative.as_posix(),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists and path.is_file() else None,
                "status": "passed" if exists else "failed",
            }
        )
    return checks


def check_zip_hygiene(root: Path, zip_path: Path = DEFAULT_ZIP) -> Dict[str, Any]:
    absolute = root / zip_path
    if not absolute.exists():
        return {
            "kind": "zip_hygiene",
            "path": zip_path.as_posix(),
            "status": "failed",
            "details": "zip file is missing",
        }

    bad_names: List[str] = []
    seed_audits: List[str] = []
    text_bad: List[str] = []
    try:
        with zipfile.ZipFile(absolute) as archive:
            names = archive.namelist()
            for name in names:
                if any(token in name for token in ["__pycache__", ".pytest_cache", ".DS_Store", "/runs/"]):
                    bad_names.append(name)
                if name.endswith((".pyc", ".pyo", ".pq", ".parquet", ".npy", ".npz", ".pkl", ".pickle", ".joblib")):
                    bad_names.append(name)
                if "/outputs/outlier_pair/" in name and "_seed" in Path(name).name:
                    seed_audits.append(name)
                data = archive.read(name)
                local_user_token = ("hpu" + "4454").encode("utf-8")
                if b"/" + b"Users" + b"/" in data or local_user_token in data:
                    text_bad.append(name)
    except zipfile.BadZipFile as exc:
        return {
            "kind": "zip_hygiene",
            "path": zip_path.as_posix(),
            "status": "failed",
            "details": f"bad zip file: {exc}",
        }

    failures = bad_names + seed_audits + text_bad
    return {
        "kind": "zip_hygiene",
        "path": zip_path.as_posix(),
        "status": "passed" if not failures else "failed",
        "file_count": len(names),
        "bad_names": bad_names[:20],
        "seed_audits": seed_audits[:20],
        "text_bad": text_bad[:20],
    }


def tier_sequence(tier: str) -> List[TierSpec]:
    if tier == "all":
        return [TIER_1, TIER_2, TIER_RENDER, TIER_RENDER_ANONYMOUS, TIER_3, TIER_PACKAGE]
    return [TIER_BY_NAME[tier]]


def run_review_check(
    tier: str,
    *,
    root: Path = REPO_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout_seconds: int | None = None,
) -> Dict[str, Any]:
    root = root.resolve()
    output_dir = (root / output_dir).resolve() if not output_dir.is_absolute() else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    specs = tier_sequence(tier)

    commands: List[Dict[str, Any]] = []
    artifact_checks: List[Dict[str, Any]] = []
    for spec in specs:
        for command in spec.commands:
            result = run_command(command, cwd=root, timeout_seconds=timeout_seconds)
            result["tier"] = spec.name
            commands.append(result)
        for check in check_expected_paths(root, spec.expected_paths):
            check["tier"] = spec.name
            artifact_checks.append(check)
        if spec.run_zip_hygiene:
            check = check_zip_hygiene(root, spec.zip_path)
            check["tier"] = spec.name
            artifact_checks.append(check)

    passed = all(command["status"] == "passed" for command in commands) and all(
        check["status"] == "passed" for check in artifact_checks
    )
    report = {
        "tier": tier,
        "started_utc": started,
        "finished_utc": _utc_now(),
        "overall_status": "passed" if passed else "failed",
        "tiers": [{"name": spec.name, "description": spec.description} for spec in specs],
        "commands": commands,
        "artifact_checks": artifact_checks,
    }
    write_reports(report, output_dir)
    return report


def write_reports(report: Dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    json_path = output_dir / "review_check_report.json"
    md_path = output_dir / "review_check_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def render_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# RANC-ContractNet Review Check Report",
        "",
        f"- Tier request: `{report['tier']}`",
        f"- Overall status: **{report['overall_status']}**",
        f"- Started UTC: {report['started_utc']}",
        f"- Finished UTC: {report['finished_utc']}",
        "",
        "## Tiers",
        "",
    ]
    for tier in report["tiers"]:
        lines.append(f"- `{tier['name']}`: {tier['description']}")

    lines.extend(["", "## Commands", ""])
    if report["commands"]:
        lines.extend(
            [
                "| Tier | Name | Status | Seconds | Return code | Command |",
                "| --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for command in report["commands"]:
            returncode = "" if command["returncode"] is None else str(command["returncode"])
            lines.append(
                f"| `{command['tier']}` | {command['name']} | {command['status']} | "
                f"{command['duration_seconds']:.3f} | {returncode} | `{command['command_display']}` |"
            )
    else:
        lines.append("No commands were configured.")

    lines.extend(["", "## Artifact Checks", ""])
    if report["artifact_checks"]:
        lines.extend(["| Tier | Kind | Path | Status | Details |", "| --- | --- | --- | --- | --- |"])
        for check in report["artifact_checks"]:
            detail = ""
            if check.get("kind") == "path":
                detail = f"exists={check.get('exists')}, bytes={check.get('size_bytes')}"
            elif check.get("kind") == "zip_hygiene":
                detail = f"files={check.get('file_count', '')}"
                failures = sum(len(check.get(key, [])) for key in ("bad_names", "seed_audits", "text_bad"))
                if failures:
                    detail += f", failures={failures}"
            lines.append(
                f"| `{check['tier']}` | {check['kind']} | `{check['path']}` | "
                f"{check['status']} | {detail} |"
            )
    else:
        lines.append("No artifact checks were configured.")

    lines.extend(["", "## Command Output Tails", ""])
    for command in report["commands"]:
        lines.append(f"### {command['tier']} / {command['name']}")
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
    parser.add_argument(
        "--tier",
        choices=[
            "1",
            "2",
            "3",
            "render",
            "render_anonymous",
            "dryrun",
            "dryrun_anonymous",
            "package",
            "package_anonymous",
            "all",
        ],
        default="1",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    args = parser.parse_args()

    report = run_review_check(
        args.tier,
        root=REPO_ROOT,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
    )
    print(args.output_dir / "review_check_report.md")
    print(args.output_dir / "review_check_report.json")
    print(f"status={report['overall_status']}")
    raise SystemExit(0 if report["overall_status"] == "passed" else 1)


if __name__ == "__main__":
    main()
