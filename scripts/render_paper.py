"""Compile the NeurIPS paper when a TeX engine is available.

The script is intentionally reviewer-friendly: missing TeX is reported as a
skipped optional check, while a discovered TeX engine that fails to compile the
paper is reported as a failed render check.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("outputs/paper_render")
PAPER_DIR = Path("paper/neurips2027")
MAIN_TEX = "main.tex"
ANONYMOUS_TEX = "main_anonymous.tex"
STYLE_FILE = "neurips_2027.sty"
VARIANT_SOURCES = {
    "identified": MAIN_TEX,
    "anonymous": ANONYMOUS_TEX,
}
VARIANT_PDFS = {
    "identified": "main.pdf",
    "anonymous": "main_anonymous.pdf",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _redact(text: str, root: Path) -> str:
    redacted = text.replace(root.resolve().as_posix(), "<REPO_ROOT>")
    redacted = redacted.replace("/" + "Users" + "/", "/<USER_HOME>/")
    redacted = redacted.replace("hpu" + "4454", "<USER>")
    return redacted


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return _redact(path.as_posix(), root)


def _which(names: Sequence[str]) -> Dict[str, str | None]:
    return {name: shutil.which(name) for name in names}


def _style_status(paper_dir: Path) -> Dict[str, Any]:
    style_path = paper_dir / STYLE_FILE
    available = style_path.exists()
    return {
        "style_file": STYLE_FILE,
        "style_mode": "official_neurips_2027" if available else "fallback_article",
        "style_file_present": available,
        "style_path": (PAPER_DIR / STYLE_FILE).as_posix(),
    }


def _run_command(argv: Sequence[str], *, cwd: Path, root: Path, timeout_seconds: int | None) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        duration = time.perf_counter() - start
        return {
            "command": [_redact(str(part), root) for part in argv],
            "command_display": " ".join(_redact(str(part), root) for part in argv),
            "returncode": completed.returncode,
            "duration_seconds": round(duration, 3),
            "status": "passed" if completed.returncode == 0 else "failed",
            "stdout_tail": _tail(_redact(completed.stdout, root)),
            "stderr_tail": _tail(_redact(completed.stderr, root)),
        }
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return {
            "command": [_redact(str(part), root) for part in argv],
            "command_display": " ".join(_redact(str(part), root) for part in argv),
            "returncode": None,
            "duration_seconds": round(duration, 3),
            "status": "failed",
            "stdout_tail": _tail(_redact(stdout, root)),
            "stderr_tail": _tail(_redact(stderr + f"\nTimed out after {timeout_seconds} seconds.", root)),
        }


def _latexmk_commands(
    tool_paths: Dict[str, str | None], build_dir: Path, source_tex: str
) -> tuple[str, List[List[str]]] | None:
    latexmk = tool_paths.get("latexmk")
    if not latexmk:
        return None
    return (
        "latexmk",
        [
            [
                latexmk,
                "-pdf",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                f"-outdir={build_dir}",
                source_tex,
            ]
        ],
    )


def _tectonic_commands(
    tool_paths: Dict[str, str | None], build_dir: Path, source_tex: str
) -> tuple[str, List[List[str]]] | None:
    tectonic = tool_paths.get("tectonic")
    if not tectonic:
        return None
    return (
        "tectonic",
        [
            [
                tectonic,
                "--keep-logs",
                "--keep-intermediates",
                "--outdir",
                str(build_dir),
                source_tex,
            ]
        ],
    )


def _pdflatex_commands(
    tool_paths: Dict[str, str | None], build_dir: Path, source_tex: str
) -> tuple[str, List[List[str]]] | None:
    pdflatex = tool_paths.get("pdflatex")
    bibtex = tool_paths.get("bibtex")
    if not pdflatex or not bibtex:
        return None
    aux_stem = build_dir / Path(source_tex).stem
    pdflatex_cmd = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        f"-output-directory={build_dir}",
        source_tex,
    ]
    return (
        "pdflatex+bibtex",
        [
            pdflatex_cmd,
            [bibtex, str(aux_stem)],
            pdflatex_cmd,
            pdflatex_cmd,
        ],
    )


def _select_commands(
    tool_paths: Dict[str, str | None], build_dir: Path, source_tex: str
) -> tuple[str, List[List[str]]] | None:
    for selector in (_latexmk_commands, _tectonic_commands, _pdflatex_commands):
        selected = selector(tool_paths, build_dir, source_tex)
        if selected is not None:
            return selected
    return None


def _render_pages(
    pdf_path: Path,
    *,
    output_dir: Path,
    variant: str,
    root: Path,
    timeout_seconds: int | None,
) -> Dict[str, Any]:
    pdftoppm = shutil.which("pdftoppm")
    if pdftoppm:
        prefix = output_dir / ("main_page" if variant == "identified" else f"main_{variant}_page")
        result = _run_command(
            [pdftoppm, "-png", "-f", "1", "-l", "3", str(pdf_path), str(prefix)],
            cwd=root,
            root=root,
            timeout_seconds=timeout_seconds,
        )
        page_glob = "main_page-*.png" if variant == "identified" else f"main_{variant}_page-*.png"
        pages = sorted(output_dir.glob(page_glob))
        return {
            "status": "passed" if result["status"] == "passed" else "failed",
            "command": result,
            "pages": [_rel(path, root) for path in pages],
        }

    qlmanage = shutil.which("qlmanage")
    if qlmanage:
        preview_dir = output_dir / ("preview" if variant == "identified" else f"preview_{variant}")
        preview_dir.mkdir(parents=True, exist_ok=True)
        result = _run_command(
            [qlmanage, "-t", "-s", "1400", "-o", str(preview_dir), str(pdf_path)],
            cwd=root,
            root=root,
            timeout_seconds=timeout_seconds,
        )
        pages = sorted(preview_dir.glob("*.png"))
        return {
            "status": "passed" if result["status"] == "passed" and pages else "failed",
            "command": result,
            "pages": [_rel(path, root) for path in pages],
        }

    return {
        "status": "skipped",
        "reason": "Neither pdftoppm nor qlmanage was found on PATH; PNG page rendering was skipped.",
        "pages": [],
    }


def run_render_check(
    *,
    root: Path = REPO_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout_seconds: int | None = 120,
    variant: str = "identified",
) -> Dict[str, Any]:
    root = root.resolve()
    if variant not in VARIANT_SOURCES:
        raise ValueError(f"Unknown paper variant {variant!r}; expected one of {sorted(VARIANT_SOURCES)}.")
    output_dir = (root / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    build_dir = output_dir / "build" / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    started = _utc_now()
    paper_dir = root / PAPER_DIR
    source_tex = VARIANT_SOURCES[variant]
    main_tex = paper_dir / source_tex
    tool_paths = _which(["latexmk", "tectonic", "pdflatex", "bibtex", "pdftoppm", "qlmanage"])
    style_status = _style_status(paper_dir)

    if not main_tex.exists():
        report = {
            "status": "failed",
            "started_utc": started,
            "finished_utc": _utc_now(),
            "variant": variant,
            "source_tex": source_tex,
            "engine": None,
            "reason": f"Missing paper source: {_rel(main_tex, root)}",
            "tool_paths": {name: bool(path) for name, path in tool_paths.items()},
            **style_status,
            "commands": [],
            "pdf_path": None,
            "page_images": [],
        }
        write_reports(report, output_dir)
        return report

    selected = _select_commands(tool_paths, build_dir, source_tex)
    if selected is None:
        report = {
            "status": "skipped",
            "started_utc": started,
            "finished_utc": _utc_now(),
            "variant": variant,
            "source_tex": source_tex,
            "engine": None,
            "reason": "No TeX engine found on PATH. Install latexmk, tectonic, or pdflatex+bibtex to render the PDF.",
            "tool_paths": {name: bool(path) for name, path in tool_paths.items()},
            **style_status,
            "commands": [],
            "pdf_path": None,
            "page_images": [],
        }
        write_reports(report, output_dir)
        return report

    engine, commands = selected
    command_results = []
    for command in commands:
        result = _run_command(command, cwd=paper_dir, root=root, timeout_seconds=timeout_seconds)
        command_results.append(result)
        if result["status"] != "passed":
            break

    pdf_candidate = build_dir / f"{Path(source_tex).stem}.pdf"
    pdf_path = output_dir / VARIANT_PDFS[variant]
    if pdf_candidate.exists():
        shutil.copy2(pdf_candidate, pdf_path)

    status = "passed" if command_results and all(item["status"] == "passed" for item in command_results) and pdf_path.exists() else "failed"
    render_result: Dict[str, Any] = {"status": "skipped", "reason": "PDF was not produced.", "pages": []}
    if status == "passed":
        render_result = _render_pages(
            pdf_path,
            output_dir=output_dir,
            variant=variant,
            root=root,
            timeout_seconds=timeout_seconds,
        )

    report = {
        "status": status,
        "started_utc": started,
        "finished_utc": _utc_now(),
        "variant": variant,
        "source_tex": source_tex,
        "engine": engine,
        "reason": "" if status == "passed" else "TeX command failed or did not produce main.pdf.",
        "tool_paths": {name: bool(path) for name, path in tool_paths.items()},
        **style_status,
        "commands": command_results,
        "pdf_path": _rel(pdf_path, root) if pdf_path.exists() else None,
        "page_render": render_result,
        "page_images": render_result.get("pages", []),
    }
    write_reports(report, output_dir)
    return report


def render_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# RANC-ContractNet Paper Render Report",
        "",
        f"- Status: **{report['status']}**",
        f"- Variant: `{report.get('variant', 'identified')}`",
        f"- Source TeX: `{report.get('source_tex', MAIN_TEX)}`",
        f"- Engine: `{report.get('engine') or 'none'}`",
        f"- Style mode: `{report.get('style_mode', 'unknown')}`",
        f"- Style file present: `{report.get('style_file_present', False)}`",
        f"- Started UTC: {report['started_utc']}",
        f"- Finished UTC: {report['finished_utc']}",
    ]
    if report.get("reason"):
        lines.append(f"- Reason: {report['reason']}")
    if report.get("pdf_path"):
        lines.append(f"- PDF: `{report['pdf_path']}`")
    if report.get("page_images"):
        lines.append(f"- Rendered page images: {', '.join(f'`{page}`' for page in report['page_images'])}")

    lines.extend(["", "## Tool Availability", ""])
    for name, available in sorted(report.get("tool_paths", {}).items()):
        lines.append(f"- `{name}`: {'available' if available else 'missing'}")

    lines.extend(["", "## Commands", ""])
    commands = report.get("commands", [])
    if not commands:
        lines.append("No TeX commands were run.")
    else:
        for index, command in enumerate(commands, start=1):
            lines.append(f"### Command {index}")
            lines.append("")
            lines.append(f"- Status: `{command['status']}`")
            lines.append(f"- Return code: `{command['returncode']}`")
            lines.append(f"- Command: `{command['command_display']}`")
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

    page_render = report.get("page_render")
    if page_render:
        lines.extend(["", "## Page Rendering", ""])
        lines.append(f"- Status: `{page_render.get('status')}`")
        if page_render.get("reason"):
            lines.append(f"- Reason: {page_render['reason']}")

    return "\n".join(lines).rstrip() + "\n"


def write_reports(report: Dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    variant = report.get("variant", "identified")
    suffix = "" if variant == "identified" else f"_{variant}"
    json_path = output_dir / f"paper_render_report{suffix}.json"
    md_path = output_dir / f"paper_render_report{suffix}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--variant", choices=sorted(VARIANT_SOURCES), default="identified")
    args = parser.parse_args()
    report = run_render_check(
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        variant=args.variant,
    )
    suffix = "" if args.variant == "identified" else f"_{args.variant}"
    print(args.output_dir / f"paper_render_report{suffix}.md")
    print(args.output_dir / f"paper_render_report{suffix}.json")
    print(f"status={report['status']}")
    raise SystemExit(1 if report["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
