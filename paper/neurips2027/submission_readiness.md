# Submission Readiness and Remaining Gap Map

This document is the project preflight checklist for turning RANC-ContractNet
from a working research artifact into a submission package. It separates what is
already locally validated from what still needs broader evidence.

## Current Status

Status: artifact-complete local draft.

The repository currently has:

- A Python package with sklearn-compatible contract compilation, serialization,
  audit reports, falsification tests, sparse safeguards, temporal drift checks,
  and exploratory torch activation-policy adapters.
- Config-driven experiment runners for paired outlier worlds, sparse checks,
  temporal drift, ablations, synthetic benchmarks, OpenML/UCI public runs, and
  small neural pilots.
- A paper draft with identified and anonymous render paths, paper-facing tables,
  a generated visual summary figure, supplementary notes, reproducibility notes,
  claims-boundary text, and artifact-evaluation guidance.
- Reviewer-style package, anonymous-package, and extracted-bundle dry-run paths.

This is not yet a final conference submission. It is a strong local artifact and
preprint candidate once the remaining evidence and review items below are
handled.

## Claim Boundary

The supported central claim is:

> RANC-ContractNet compiles normalization policies from declared invariance
> contracts and makes the resulting decision auditable, replayable, and
> falsifiable.

The current evidence supports:

- Correct versus wrong contracts change the legal normalization policy in paired
  outlier signal/noise worlds.
- Correct contracts reduce targeted semantic failure metrics in the paired
  outlier benchmark.
- Sparse and temporal checks expose representation and split-safety evidence.
- Public OpenML/UCI runs provide scoped public-data execution evidence with
  inclusion/exclusion metadata and paired deltas.
- The artifact can be rendered, packaged, anonymized, extracted, and rerun
  locally.

The current evidence does not support:

- Universal predictive dominance over all scalers or validation selectors.
- Replacing validation search when the sole goal is raw predictive score.
- Treating neural activation-policy results as central rather than exploratory.
- Claiming that RANC discovers true domain semantics without user contracts or
  supervised opt-in.

## When To Use RANC

Use RANC when normalization must be:

- Auditable: a reviewer needs to know why a transform was legal.
- Reproducible: fitted statistics, rejected candidates, and downgrades must be
  serialized.
- Split-safe: train-only fitting and temporal discipline matter.
- Representation-safe: sparse zeros, signs, monotonicity, or inverse behavior
  are part of the task contract.
- Semantically constrained: rare extremes may be corruption in one setting and
  signal in another.
- Regulated or high-review: healthcare, finance, scientific tabular pipelines,
  risk models, compliance workflows, and production ML systems with preprocessing
  audits.

Do not position RANC as the default choice when:

- The only objective is maximum held-out predictive score.
- The user cannot state any meaningful invariance or representation contract.
- An AutoML/scaler selector is acceptable and auditability is not required.
- Domain semantics are unknown and no supervised opt-in is allowed.

## Validated Local Commands

These commands define the local readiness bar:

```bash
python3 -m pytest
python3 -m compileall src experiments scripts tests
python3 scripts/generate_paper_figures.py
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier render
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier render_anonymous
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier package
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier package_anonymous
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier dryrun
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier dryrun_anonymous
```

Passing these commands means the local code, paper render, figures, packages,
anonymous package, and extracted-bundle reviewer simulation are internally
consistent. It does not mean the research evidence is broad enough for a final
main-track claim.

## Preprint Readiness

A serious preprint is realistic after:

- The current validation commands pass from a clean checkout or extracted
  bundle.
- The anonymous PDF and anonymous zip scan clean for author, email, affiliation,
  role-title, local path, and local validation-report leakage.
- The paper avoids universal predictive-dominance claims.
- The visual summary figure and main result tables are regenerated from the
  documented outputs.
- The README, reproducibility notes, claims boundary, artifact guide, and
  supplementary notes agree on what is central and what is exploratory.
- At least one independent reader checks the paper for clarity, claim scope,
  and missing definitions.

## NeurIPS-Style Submission Gap

For a stronger conference submission, the remaining research work is:

- Broaden the public benchmark suite with a preregistered task list, fixed
  exclusions, and retained failure metadata.
- Add more baseline families: common sklearn scalers, validation-selected
  selectors, leakage-prone anti-controls, and task-specific normalizers where
  relevant.
- Add domain-style case studies where contract semantics are natural, especially
  sparse features, temporal prediction, and rare-event tabular workflows.
- Strengthen statistical reporting with preregistered metrics, paired tests,
  confidence intervals, and full seed/task tables.
- Add a reviewer-visible adoption story: API example, audit-report example,
  integration into sklearn `Pipeline`, and guidance for writing contracts.
- Keep the neural activation-policy path clearly exploratory unless larger
  neural experiments justify a central claim.
- Re-run the final artifact under the official conference style and checklist
  once the target venue releases its current requirements.

## Remaining Engineering Gap

The main engineering tasks before a public release are:

- Initialize version control and tag a reproducible artifact snapshot.
- Add continuous integration for tests, compile checks, figure generation, and
  package hygiene.
- Add a minimal examples directory with one sparse, one temporal, and one
  rare-signal walkthrough.
- Add a changelog and release notes.
- Decide whether generated PDFs and figures are committed, regenerated in CI, or
  shipped only in release artifacts.
- Review package metadata, license, and repository URLs before public upload.

## Reviewer Risk Register

Expected reviewer objection: "Is this just AutoML?"

Response: No. AutoML optimizes held-out score over pipeline choices. RANC
compiles the least complex legal policy under declared contracts and audit cost,
without using validation/test outcomes for scaler choice.

Expected reviewer objection: "What if the contract is wrong?"

Response: RANC can faithfully preserve the wrong structure. The contribution is
that this failure is inspectable through wrong-contract controls, ledger rows,
downgrades, and targeted metrics instead of hidden inside a selected scaler name.

Expected reviewer objection: "Why does the validation selector sometimes win?"

Response: The selector optimizes a different target: held-out predictive score.
That result is informative but not a refutation of contract legality,
auditability, or split safety.

Expected reviewer objection: "Where should this be used?"

Response: RANC is most useful when preprocessing must be semantically legal,
auditable, and reproducible, especially sparse, temporal, rare-signal, regulated,
or high-review tabular pipelines.

Expected reviewer objection: "Where is the broad evidence?"

Response: The current artifact has decisive paired semantic tests and scoped
public-data evidence. A final conference submission should broaden the
preregistered public suite before making broad empirical claims.

## Finish Criteria

Treat the project as preprint-ready when all local validation commands pass and
the preprint-readiness items above are checked.

Treat the project as conference-submission-ready only after the broader
benchmark, external review, official style/checklist, and release-engineering
gaps are closed.
