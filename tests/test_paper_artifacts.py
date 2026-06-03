from pathlib import Path


def _compact(text: str) -> str:
    return " ".join(text.split())


def test_paper_scaffold_exists():
    root = Path("paper/neurips2027")
    for name in [
        "main.tex",
        "checklist.md",
        "supplementary.md",
        "reproducibility.md",
        "references.bib",
        "claims_boundary.md",
        "submission_readiness.md",
        "release_snapshot.md",
        "artifact_eval.md",
        "STYLE_STATUS.md",
    ]:
        path = root / name
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip()


def test_main_paper_wires_first_decisive_result():
    text = Path("paper/neurips2027/main.tex").read_text(encoding="utf-8")
    compact_text = _compact(text)
    assert "RANC-ContractNet: Auditable Normalization as Invariance Compilation" in text
    identified_author = "Harshad Hemant Patil" in text
    anonymous_author = "Anonymous " + "Author" in text
    assert identified_author or anonymous_author
    if identified_author:
        assert "Lead Data Scientist, Springer Nature" in text
        assert "hhpatil001@gmail.com" in text
    else:
        assert "Anonymous " + "Institution" in text
        assert "anonymous" + "@example.org" in text
    assert "Anonymous " + "Authors" not in text
    assert "\\section{Related Work}" in text
    for phrase in [
        "Normalization is often treated as a preprocessing hyperparameter",
        "Validation-score scaler search can hide illegal choices",
        "RANC-ContractNet is not a new scaler",
        "invariance compilation rather than AutoML scaler search",
        "implemented checkers, not universal predictive dominance",
        "Most machine-learning pipelines name a normalizer but not the contract",
        "This framing is most useful when preprocessing decisions carry scientific",
        "It is less appropriate as a default replacement for validation search",
        "a legal and auditable compiler for existing primitives",
        "Those tools answer the",
        "is this transform licensed by the task contract",
        "which normalization policies are permissible before validation performance is consulted",
        "validation-selected preprocessing a comparator rather than the mechanism",
        "fit discipline alone is not a normalization contract",
        "The audit target is deliberately narrower than a dataset sheet or model card",
        "the normalization decision itself",
        "axis choice, centering choice, saturation",
        "neural claims should be read as interface coverage",
        "Labels may be used only when",
        "finite deterministic candidate set",
        "emits either a selected fitted policy",
        "The guarantee is relative to the declared contract",
        "The claim is auditable legality under declared invariances",
        "no-op is a valid compiled outcome rather than a failed search",
        "Definition 1 (Regime card)",
        "Definition 2 (Invariance contract)",
        "Definition 3 (Legal normalization policy)",
        "Legal does not mean predictively optimal or semantically correct",
        "Definition 4 (Signal-risk ledger)",
        "Definition 5 (Falsification acceptance rule)",
        "numeric falsification tests certify only their deterministic audit support",
        "\\mathcal{A}_g",
        "Only policies in $\\mathcal{A}_g$ are eligible for tie breaking",
        "\\label{tab:contract-evidence}",
        "Contract clauses map directly to compiler behavior",
        "Train-only fitting",
        "Sparse no-densification",
        "Outlier-as-signal",
        "Compiler behavior and audit/falsification evidence",
        "RANC-ContractNet is not AutoML under another name",
        "Wrong contracts can preserve the wrong structure",
        "Proposition 1 (relative contract soundness)",
        "relative to the declared contract, candidate set, and checker support",
        "does not prove semantic truth, downstream optimality",
        "User preferences can break ties only inside the survivor set",
        "The audit is a typed JSON-serializable object",
        "Replay means that a serialized transformer can be reconstructed",
        "The paper makes five contributions",
        "uses hard clauses rather than validation outcomes",
        "packaging hygiene",
        "The experiments are organized as a three-level evidence chain",
        "Level 1 tests contract causality",
        "Level 2 tests safety clauses",
        "Level 3 tests public-data execution",
        "\\label{tab:claims-tested}",
        "Evidence map for the contract-compilation claim",
        "Rows are ordered by",
        "no row is a universal scaler leaderboard",
        "\\label{fig:ranc-visual-summary}",
        "Visual summary of the method and two main quantitative summaries",
        "Panel A is the mechanism",
        "Panel B is the decisive semantic test",
        "Panel C is the scoped public-data check",
        "figures/ranc_visual_summary.pdf",
        "Paired outlier worlds",
        "Artifact checks",
        "clean extracted-bundle dry runs",
        "\\subsection{Level 1: paired outlier signal/noise benchmark}",
        "This is the decisive benchmark because it isolates the variable RANC claims to control",
        "If RANC is only a disguised validation-score scaler search",
        "Decisive semantic intervention on paired outlier worlds",
        "\\subsection{Level 2: sparse and temporal safety checks}",
        "Artifact-level safety evidence for hard clauses",
        "signed OpenML/UCI deltas show where RANC helps or loses",
        "The second evidence block tests hard clauses",
        "declared representation and split constraints",
        "\\subsection{Level 3: public OpenML/UCI benchmark path}",
        "These results are public-data artifact evidence, not the central contract causality test",
        "\\label{tab:safety-summary}",
        "Experimental protocol and reporting discipline",
        "seeds $0$--$29$",
        "5,000 bootstrap samples",
        "seeds $0$--$4$",
        "seeds $0$--$2$",
        "protocols, exclusion rules, PDF-render checks",
        "no cherry-picking across seeds, datasets, or exclusions",
        "\\subsection{Failure modes and detection}",
        "full checklists are supplementary",
        "\\subsection{Result narrative and falsifiability}",
        "The main empirical takeaway is not that one scaler wins everywhere",
        "paired outlier benchmark supplies the strongest evidence",
        "they are supportive but not central",
        "A selector that searches preprocessing choices by held-out outcome",
        "RANC can faithfully preserve the wrong structure",
        "wrong-contract controls, ledger rows, and degraded targeted metrics",
        "The approach would be weakened or falsified by future preregistered runs",
        "These tests are stricter than asking whether one scaler wins",
        "correct contracts no longer improve the targeted failure metric",
        "audits for completed tasks, exclusion metadata, and paired predictive deltas",
        "\\section{Conclusion}",
        "RANC-ContractNet reframes normalization as a checked compilation step",
        "does not claim to discover true domain semantics",
        "with conservative downgrades or no-ops recorded when legality cannot be established",
    ]:
        assert phrase in text or phrase in compact_text
    assert "\\label{tab:failure-modes}" not in text
    assert "\\label{tab:reviewer-questions}" not in text
    for key in [
        "pedregosa2011scikit",
        "feurer2015automl",
        "olson2016tpot",
        "ioffe2015batch",
        "ba2016layer",
        "wu2018group",
        "zhang2019rmsnorm",
        "zhu2025dyt",
        "kaufman2012leakage",
        "gebru2021datasheets",
        "mitchell2019modelcards",
    ]:
        assert key in text
    assert "Algorithm 1: RANC-ContractNet fit" in text
    assert "contract_causal_table.tex" in text
    assert "contract_delta_table.tex" in text
    assert "ablation_table.tex" in text
    assert "openml_win_loss_table.tex" in text
    assert "benchmark_table.tex" not in text
    assert "openml_table.tex" not in text
    assert "sparse_table.tex" not in text
    assert "temporal_drift_table.tex" not in text
    assert "contract correctness, not hidden" in text
    assert "Limitations and Claims Boundary" in text
    assert "not presented as a universal predictive leaderboard winner" in text
    assert "Public results should be read on two axes" in compact_text
    assert "artifact_eval.md" in Path("paper/neurips2027/reproducibility.md").read_text(
        encoding="utf-8"
    )

    outputs = Path("outputs/outlier_pair")
    assert (outputs / "contract_causal_table.tex").exists()
    assert (outputs / "contract_delta_table.tex").exists()
    assert (outputs / "contract_statistical_paragraph.md").exists()

    assert Path("outputs/sparse/sparse_table.tex").exists()
    assert Path("outputs/temporal_drift/temporal_drift_table.tex").exists()
    assert Path("outputs/ablations/ablation_table.tex").exists()
    assert Path("outputs/benchmark/benchmark_table.tex").exists()
    assert Path("outputs/openml/openml_table.tex").exists()
    assert Path("outputs/openml/openml_win_loss_table.tex").exists()
    assert Path("outputs/openml/openml_stats_paragraph.md").exists()
    assert Path("paper/neurips2027/figures/ranc_visual_summary.pdf").exists()
    assert Path("paper/neurips2027/figures/ranc_visual_summary.png").exists()

    supplementary = Path("paper/neurips2027/supplementary.md").read_text(encoding="utf-8")
    for phrase in [
        "compact safety summary table",
        "full generated sparse and temporal tables",
        "full aggregate synthetic table",
        "full task-level table",
        "outputs/sparse/sparse_table.tex",
        "outputs/temporal_drift/temporal_drift_table.tex",
        "outputs/benchmark/benchmark_table.tex",
        "outputs/openml/openml_table.tex",
    ]:
        assert phrase in supplementary


def test_related_work_bibliography_entries_exist():
    bib = Path("paper/neurips2027/references.bib").read_text(encoding="utf-8")
    for key in [
        "pedregosa2011scikit",
        "feurer2015automl",
        "olson2016tpot",
        "kaufman2012leakage",
        "yeo2000new",
        "ioffe2015batch",
        "ba2016layer",
        "wu2018group",
        "zhang2019rmsnorm",
        "zhu2025dyt",
        "gebru2021datasheets",
        "mitchell2019modelcards",
    ]:
        assert f"{{{key}," in bib


def test_paper_claims_boundary_avoids_overclaiming_language():
    root = Path("paper/neurips2027")
    checked = [
        root / "main.tex",
        root / "supplementary.md",
        root / "reproducibility.md",
        root / "claims_boundary.md",
        root / "artifact_eval.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8").lower() for path in checked)
    forbidden = [
        "state-of-the-art",
        "outperforms all baselines",
        "outperform all baselines",
    ]
    for phrase in forbidden:
        assert phrase not in combined

    claims = _compact((root / "claims_boundary.md").read_text(encoding="utf-8"))
    assert "not as a hidden validation-score scaler search procedure" in claims
    assert "predictive utility and contract/audit guarantees" in claims
    assert "Failure Modes" in claims
    assert "Underspecified contracts can lead to conservative no-op" in claims
    assert "Wrong declared semantics can make RANC preserve the wrong structure" in claims
    assert "Result Interpretation" in claims
    assert "The strongest current evidence is the paired outlier signal/noise benchmark" in claims
    assert "The contract-compilation claim would be weakened or falsified" in claims
    assert "Author Response Checklist" in claims
    assert "AutoML/scaler-selection objection" in claims
    assert "Reproducibility objection" in claims

    readiness = _compact((root / "submission_readiness.md").read_text(encoding="utf-8"))
    for phrase in [
        "Submission Readiness and Remaining Gap Map",
        "artifact-complete local draft",
        "RANC-ContractNet compiles normalization policies from declared invariance",
        "contracts and makes the resulting decision auditable",
        "The current evidence does not support",
        "Universal predictive dominance",
        "When To Use RANC",
        "Do not position RANC as the default choice",
        "Validated Local Commands",
        "Preprint Readiness",
        "NeurIPS-Style Submission Gap",
        "Remaining Engineering Gap",
        "Reviewer Risk Register",
        "Finish Criteria",
        "Treat the project as preprint-ready",
        "Treat the project as conference-submission-ready",
    ]:
        assert phrase in readiness

    release_snapshot = _compact((root / "release_snapshot.md").read_text(encoding="utf-8"))
    for phrase in [
        "artifact-local-draft-2026-06-03",
        "artifact-complete local draft",
        "Included Scope",
        "Excluded Local Noise",
        "Validation Gate",
        "Claim Boundary",
        "does not claim universal predictive dominance",
        "Identity Modes",
        "Harshad Hemant Patil",
        "hhpatil001@gmail.com",
    ]:
        assert phrase in release_snapshot

    supplementary = (root / "supplementary.md").read_text(encoding="utf-8")
    for phrase in [
        "Experimental Protocol and Reporting Discipline",
        "enabled in `experiments/configs/openml_public.yaml`",
        "Temporal drift tasks use the prefix 70% of the sequence",
        "Failure Modes and Artifact Signals",
        "Underspecified contract",
        "Wrong declared semantics",
        "Rare-signal ambiguity",
        "Predictive objective mismatch",
        "Drift without adaptation",
        "Legal but unhelpful transform",
        "Exploratory neural path",
        "Likely Reviewer Questions",
        "Is this just AutoML or scaler selection?",
        "What if the contract is wrong?",
        "Where is leakage protection?",
        "Can the numbers be reproduced?",
    ]:
        assert phrase in supplementary

    artifact_eval = (root / "artifact_eval.md").read_text(encoding="utf-8")
    for phrase in [
        "Experimental Protocol Lock",
        "Failure Modes to Inspect",
        "no-op reasons and downgrades",
        "wrong-contract controls and ledger rows",
        "experiments/configs/*.yaml",
        "included/excluded/failed task status",
        "Tier 1: Fast CPU Validation",
        "Optional Paper PDF Render Check",
        "scripts/review_check.py --tier render",
        "scripts/generate_paper_figures.py",
        "paper/neurips2027/figures/ranc_visual_summary.pdf",
        "paper/neurips2027/submission_readiness.md",
        "Extracted Bundle Dry Run",
        "scripts/review_check.py --tier dryrun",
        "outputs/artifact_dry_run/extracted_bundle_report.md",
        "not inserted back into the zip",
        "outputs/paper_render/paper_render_report.md",
        "Style mode",
        "status is `skipped`",
        "Tier 2: Synthetic Paper Artifact Regeneration",
        "Tier 3: Network-Dependent OpenML/UCI Regeneration",
        "Tier 4: Optional Torch/Neural Pilot",
        "scripts/review_check.py --tier 1",
        "scripts/review_check.py --tier all",
        "Pass criteria",
    ]:
        assert phrase in artifact_eval

    reproducibility = (root / "reproducibility.md").read_text(encoding="utf-8")
    for phrase in [
        "Experimental Protocol",
        "seeds `0` through `29`",
        "stats_random_state: 12345",
        "outputs/openml/openml_task_metadata.csv",
        "Render: `python3 scripts/review_check.py --tier render`",
        "Figures: `python3 scripts/generate_paper_figures.py`",
        "Extracted bundle dry run: `python3 scripts/review_check.py --tier dryrun`",
        "outputs/paper_render/paper_render_report.json",
        "outputs/artifact_dry_run/extracted_bundle_report.md/json",
        "Reporting discipline",
    ]:
        assert phrase in reproducibility

    style_status = (root / "STYLE_STATUS.md").read_text(encoding="utf-8")
    for phrase in [
        "fallback `article`",
        "not a claim of official NeurIPS 2027 style compliance",
        "paper/neurips2027/neurips_2027.sty",
        "official_neurips_2027",
        "Recheck page count under the official style",
    ]:
        assert phrase in style_status

    supplementary = (root / "supplementary.md").read_text(encoding="utf-8")
    for phrase in [
        "python3 scripts/review_check.py --tier render",
        "outputs/paper_render/paper_render_report.md",
        "skipped` when no complete TeX toolchain is available",
    ]:
        assert phrase in supplementary
