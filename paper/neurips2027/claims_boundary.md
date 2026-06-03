# Claims Boundary

RANC-ContractNet should be evaluated as an auditable contract compiler, not as a
hidden validation-score scaler search procedure.

The current evidence supports these claims:

- Declared contracts change the selected policy in paired outlier signal/noise
  worlds, and the correct contract reduces the scenario-specific failure mode.
- The sklearn transformer records train-only fitted statistics, rejected
  candidates, downgrades, ledger rows, drift monitors, and falsification results.
- Sparse and temporal smoke tests preserve the representation and split
  discipline expected from the declared contracts.
- Public OpenML/UCI runs produce reproducible metrics, task metadata, exclusion
  logs, audit artifacts, and paired RANC-baseline deltas.

The current evidence does not support these claims:

- RANC is not presented as a universal predictive leaderboard winner.
- RANC is not a replacement for validation search when the only objective is
  maximizing downstream score.
- The neural activation-policy path is exploratory until larger neural
  experiments justify central claims.
- Public benchmark conclusions must be tied to the documented suite, seeds,
  excluded datasets, and any transient fetch/run failures.

The validation selector is allowed to optimize held-out score over a menu of
preprocessors. RANC deliberately does not use that signal to choose a scaler.
Therefore, if the selector wins on raw public predictive score, that is evidence
for a different objective rather than a refutation of the contract-compilation
claim. The fair comparison should report both axes: predictive utility and
contract/audit guarantees.

## Intended Use

Use RANC when normalization choices carry semantics beyond raw predictive score:
sparse implicit zeros must remain meaningful; temporal or grouped splits make
future statistics illegal; rare extremes may be signal rather than
contamination; inverse transforms are required for reporting; or deployment
requires a replayable preprocessing audit.

Do not present RANC as a drop-in promise of higher accuracy, a substitute for
domain knowledge, or evidence that a user-supplied contract is true. If the only
objective is held-out predictive score and no invariance, split, inverse, or
representation constraint is being asserted, validation search is the more
direct tool.

The final paper should use restrained language: RANC compiles normalization
policies from declared invariance contracts, exposes the resulting audit trail,
and can be tested for contract compliance. It should avoid claiming universal
predictive dominance unless a much larger preregistered benchmark suite supports
that claim.

## Result Interpretation

The strongest current evidence is the paired outlier signal/noise benchmark,
because it changes declared semantics while holding paired data structure fixed.
Sparse and temporal checks are engineering-safety evidence. OpenML/UCI results
are public-data artifact evidence and should be read with the exclusion log,
paired deltas, and audit pass rates.

The validation selector can win raw public predictive score because it searches
preprocessing choices by held-out outcome. That result is not a contradiction of
RANC unless the paper claims raw-score optimization as the central objective.

The contract-compilation claim would be weakened or falsified if correct
contracts fail to improve targeted failure metrics over wrong-contract controls,
if audits pass while transformed data violates train-only/sparse/drift/inverse
constraints, or if the documented configs cannot reproduce the reported tables
and audit reports.

## Author Response Checklist

- AutoML/scaler-selection objection: RANC is contract-constrained compilation,
  not validation-score scaler search.
- Wrong-contract objection: wrong semantics can produce wrong preservation, and
  paired wrong-contract controls plus ledger rows are the detection path.
- Validation-selector objection: raw predictive score and contract/audit
  compliance are separate axes.
- Leakage objection: train-only fit discipline is enforced through sklearn
  pipelines, tests, audit fields, and temporal prefix metadata.
- Neural-results objection: neural adapters are exploratory and are not central
  claims.
- Reproducibility objection: use tiered reviewer commands, expected artifact
  checks, and the packaged SHA256 bundle.

## Failure Modes

- Underspecified contracts can lead to conservative no-op or weaker legal
  policies; reviewers should inspect downgrades, rejected candidates, and no-op
  reasons.
- Wrong declared semantics can make RANC preserve the wrong structure; paired
  wrong-contract controls and ledger rows are the intended detection path.
- Rare-signal handling requires domain contracts or supervised opt-in; otherwise
  the compiler should not infer that extremes are meaningful.
- Drift monitors expose shift but do not automatically solve adaptation,
  retraining, or deployment policy.
- A legal normalization policy can still be unhelpful for a downstream model, so
  results should report both predictive metrics and audit diagnostics.
- Neural activation-policy results remain exploratory until backed by a larger
  neural benchmark.
- The method would need stronger evidence if audits passed while transformed
  data violated hard clauses or if correct contracts failed to outperform
  wrong-contract controls on targeted semantic stress tests.
