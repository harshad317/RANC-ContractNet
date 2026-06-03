"""Audit export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from ranc_contractnet.schemas import AuditReport


def _coerce_report(value: Any) -> AuditReport:
    if isinstance(value, AuditReport):
        return value
    if hasattr(value, "get_audit_report"):
        return value.get_audit_report()
    raise TypeError("Expected an AuditReport or fitted object with get_audit_report().")


def _markdown_report(report: AuditReport) -> str:
    policy_rows = "\n".join(
        f"| {p.feature_name} | {p.policy_type} | {p.complexity} | {p.downgrade_reason or ''} |"
        for p in report.policies
    )
    failures = [r for r in report.falsification_results if not r.passed]
    failure_rows = "\n".join(
        f"| {r.feature_name} | {r.policy_type} | {r.test_name} | {r.hard_clause} | {r.message} |"
        for r in failures
    )
    ledger_rows = "\n".join(
        f"| {row.feature_name} | {row.proposed_nuisance} | {row.possible_signal_removed} | {row.mitigation} |"
        for row in report.ledger_rows
    )
    return f"""# RANC-ContractNet Audit Report

- Created at: `{report.created_at}`
- Random state: `{report.random_state}`
- Passed hard falsification: `{report.passed}`
- Features: `{len(report.feature_names)}`

## Policies

| Feature | Policy | Complexity | Downgrade |
| --- | --- | ---: | --- |
{policy_rows or '|  |  |  |  |'}

## Signal Risk Ledger

| Feature | Proposed nuisance | Possible signal removed | Mitigation |
| --- | --- | --- | --- |
{ledger_rows or '|  |  |  |  |'}

## Falsification Failures

| Feature | Policy | Test | Hard clause | Message |
| --- | --- | --- | --- | --- |
{failure_rows or '|  |  |  |  |  |'}

## Warnings

{chr(10).join(f'- {warning}' for warning in report.warnings) or '- None'}
"""


def export_report(value: Union[AuditReport, Any], path: Union[str, Path], format: str = "json") -> Path:
    """Export an audit report as JSON or Markdown."""

    report = _coerce_report(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = format.lower()
    if normalized == "json":
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    elif normalized in {"md", "markdown"}:
        path.write_text(_markdown_report(report), encoding="utf-8")
    else:
        raise ValueError("format must be 'json' or 'md'.")
    return path

