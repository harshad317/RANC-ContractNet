# Contributing

RANC-ContractNet is currently a research artifact for a paper draft. Contributions
should preserve reproducibility, claim discipline, and auditability.

## Development Setup

```bash
python3 -m pip install -e ".[dev,experiments]"
python3 -m pytest
```

Optional neural checks require:

```bash
python3 -m pip install -e ".[torch]"
```

## Branch Discipline

- Use `main` as the integration branch.
- Keep changes small and reviewable.
- Do not commit local caches, package zips, raw OpenML run folders, render build
  directories, or per-seed audit dumps.
- Regenerate paper-facing outputs only through the documented experiment
  commands.

## Required Checks

Before opening a pull request or cutting a new release, run:

```bash
python3 -m pytest
python3 -m compileall src experiments scripts tests
```

For paper or artifact changes, also run the relevant reviewer tiers:

```bash
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier render
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier render_anonymous
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier package
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier package_anonymous
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier dryrun
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier dryrun_anonymous
```

## Claim Discipline

Do not describe RANC-ContractNet as a universal predictive leaderboard winner.
The supported claim is narrower: the system compiles normalization policies from
declared invariance contracts and makes the resulting decisions auditable,
replayable, and falsifiable relative to implemented checkers.

## Paper Changes

Paper edits should preserve:

- the distinction between contract compilation and validation-score scaler
  search;
- the claims boundary in `paper/neurips2027/claims_boundary.md`;
- the anonymous submission path through `main_anonymous.tex`;
- reproducibility instructions in `paper/neurips2027/reproducibility.md`; and
- artifact validation expectations in `paper/neurips2027/artifact_eval.md`.
