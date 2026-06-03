# Reviewer Packet

This packet is a working author-response map for the current RANC-ContractNet
draft. It records the strongest likely objections, the current evidence, and the
paper edits needed before preprint or conference submission.

## Topline Assessment

The artifact is locally validated and the paper is now best read as an
auditable-normalization contribution, not as a predictive leaderboard paper.
The strongest acceptance path is:

1. Establish that common normalizers encode implicit invariance claims.
2. Show that RANC makes those claims explicit as contracts.
3. Show that contracts change the legal compiled policy under paired semantics.
4. Show that hard clauses produce observable audit evidence and downgrades.
5. Report public OpenML/UCI execution honestly, including selector losses.

The main risk is not implementation maturity. The main risk is reviewer belief:
reviewers must be convinced that auditable legality is a meaningful objective
separate from validation-score scaler search.

## Likely Reviewer Objections

### 1. "Is this just AutoML or scaler selection?"

Severity: high.

Current defense: The method chooses from legal candidates before validation
score is consulted. The validation selector is reported as a comparator and can
win on raw score.

Main-text requirement: Keep the distinction visible in the abstract,
introduction, method objective, experiments, and limitations. The paper must not
sound like it is hiding a weak AutoML benchmark.

Author-response line: RANC optimizes legality under declared invariance,
representation, split, inverse, and audit clauses. AutoML optimizes validation
score under a search budget. The objectives are different, so selector wins on
raw score are informative but not dispositive.

### 2. "Where is the novelty if the transforms are existing scalers?"

Severity: high.

Current defense: The novelty is the selection and audit interface: typed
contracts, regime cards, signal-risk ledgers, hard-clause rejection, deterministic
fallbacks, and falsification artifacts.

Main-text requirement: Do not present standardization, robust scaling, quantile
maps, or power transforms as new. Present the new object as the contract compiler
and its audit trail.

Author-response line: RANC does not invent a new scalar transform. It changes
the unit of review from "which scaler was fit?" to "what contract licensed this
fitted normalizer, which candidates were illegal, and what evidence can falsify
the decision?"

### 3. "What if the contract is wrong?"

Severity: high.

Current defense: Wrong contracts are explicitly modeled through paired controls,
ledger rows, and degraded targeted metrics.

Main-text requirement: The paper must keep saying that RANC exposes wrong
semantics; it does not discover true semantics automatically.

Author-response line: Wrong contracts can preserve the wrong structure. That is
why the artifact includes wrong-contract controls and failure-mode reporting.
The contribution is inspectability, not semantic omniscience.

### 4. "The strongest evidence is synthetic. Is that enough?"

Severity: high for a main-track conference; moderate for a preprint.

Current defense: The paired outlier worlds are the cleanest causal test because
data positions and rare extremes are held fixed while only declared semantics
change. OpenML/UCI results are retained as public execution evidence, not as the
central causal claim.

Main-text requirement: Preserve the evidence hierarchy. Do not imply that the
public benchmark proves broad predictive dominance.

Author-response line: The central claim is semantic legality under contracts, so
the decisive test is a controlled semantic intervention. A final conference
submission should still broaden preregistered public/domain benchmarks before
claiming wide empirical advantage.

### 5. "Why does the validation selector beat RANC on public tasks?"

Severity: medium-high.

Current defense: The selector optimizes held-out score directly; RANC does not
use validation outcomes to choose a scaler.

Main-text requirement: Keep selector losses in the paper. Treat them as a
pressure test, not an embarrassment.

Author-response line: A score-optimizing selector should often win raw score
when no audit or semantic legality objective is imposed. That result supports
the boundary: use RANC when legality and auditability matter.

### 6. "Are the guarantees too weak because they are relative to checkers?"

Severity: medium-high.

Current defense: Proposition 1 explicitly states relative contract soundness:
the guarantee is relative to the declared contract, candidate set, and checker
support.

Main-text requirement: The limitations section should name checker coverage as a
threat to validity, especially numeric support versus global behavior.

Author-response line: The paper does not claim global semantic proof. It claims
replayable legality on stated support and recorded downgrades/no-ops when
legality cannot be established.

### 7. "Will practitioners actually use this?"

Severity: medium.

Current defense: The intended users are sparse, temporal, rare-event, regulated,
scientific, and high-review tabular pipelines where preprocessing needs an audit
trail.

Main-text requirement: Keep the adoption boundary concrete. Avoid vague claims
that everyone should replace their scaler with RANC.

Author-response line: RANC is for workflows where the preprocessing decision
must be justified, replayed, and checked. If raw held-out score is the only
objective, validation search is the direct tool.

### 8. "Are neural results central?"

Severity: low if kept scoped, high if overclaimed.

Current defense: Neural adapters are exploratory interface coverage.

Main-text requirement: Keep neural claims out of the main evidence chain unless
larger neural benchmarks are added.

Author-response line: The central contribution is the preprocessing compiler;
neural layers are an extension showing that the contract vocabulary can expose
axis, centering, saturation, and train/inference audit behavior.

## Required Main-Text Edits

Completed in this pass:

- Make the limitations section name threats to validity directly.
- Add checker coverage and evidence breadth as explicit threats.
- Keep wrong-contract and validation-selector objections in the main paper.
- Preserve the adoption boundary around audit-heavy preprocessing workflows.

Still recommended before a conference submission:

- Add at least one domain-style case study with natural contracts.
- Expand the public benchmark suite with preregistered tasks and retained
  exclusion metadata.
- Add a short API/audit-report example if page budget permits.
- Get one independent reader to review whether the novelty is clear by page 2.

## Current Decision

The paper is preprint-plausible after local validation and one independent
clarity review. It is not yet a finished NeurIPS/ICML main-track submission
because broader empirical evidence and external review are still needed.
