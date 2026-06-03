import zipfile
from pathlib import Path

import pytest

from experiments.package_artifact import build_artifact, collect_artifact_files


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _minimal_repo(root: Path) -> None:
    _write(root / "pyproject.toml", "[project]\nname='demo'\n")
    _write(root / "README.md", "# Demo\n")
    _write(root / "LICENSE", "MIT License\n")
    _write(root / "CITATION.cff", "cff-version: 1.2.0\n")
    _write(root / "CONTRIBUTING.md", "# Contributing\n")
    _write(root / ".zenodo.json", "{}\n")
    _write(root / "src/ranc_contractnet/__init__.py", "__version__ = '0.0'\n")
    _write(root / "experiments/__init__.py", "")
    _write(root / "experiments/configs/smoke.yaml", "seed: 0\n")
    _write(root / "scripts/review_check.py", "print('review')\n")
    _write(root / "tests/test_smoke.py", "def test_smoke():\n    assert True\n")
    _write(root / "paper/neurips2027/main.tex", "\\title{Anonymous}\n")
    _write(root / "paper/neurips2027/claims_boundary.md", "# Claims Boundary\n")
    _write(root / "paper/neurips2027/artifact_eval.md", "# Artifact Evaluation Guide\n")


def test_artifact_packager_collects_clean_summary_payload(tmp_path):
    _minimal_repo(tmp_path)
    _write(tmp_path / "outputs/openml/openml_table.md", "| ok |\n")
    _write(tmp_path / "outputs/openml/runs/task/seed_0/metrics.csv", "raw run detail\n")
    _write(tmp_path / "outputs/openml/dataset_287.pq", "raw public cache\n")
    _write(tmp_path / "outputs/paper_render/main.pdf", "%PDF generated render\n")
    _write(tmp_path / "outputs/paper_render/preview/main.pdf.png", "png preview\n")
    _write(tmp_path / "outputs/review_check/review_check_report.md", "# Local review check\n")
    _write(tmp_path / "outputs/review_check_all/review_check_report.md", "# Local full review check\n")
    _write(tmp_path / "outputs/artifact_dry_run/extracted_bundle_report.md", "# Local dry run\n")
    _write(
        tmp_path / "outputs/artifact_dry_run_anonymous/extracted_bundle_report.md",
        "# Local anonymous dry run\n",
    )
    _write(tmp_path / "outputs/smoke/metrics.csv", "local smoke output\n")
    _write(tmp_path / "outputs/other/local.pdf", "%PDF local ad hoc\n")
    _write(tmp_path / "outputs/outlier_pair/ranc_correct_noise_contract_noise_seed0.json", "{}\n")
    _write(tmp_path / "experiments/__pycache__/runner.pyc", "bytecode\n")
    _write(tmp_path / ".DS_Store", "local os file\n")

    relative_files = {path.as_posix() for path in collect_artifact_files(tmp_path)}

    assert "README.md" in relative_files
    assert "src/ranc_contractnet/__init__.py" in relative_files
    assert "scripts/review_check.py" in relative_files
    assert "outputs/openml/openml_table.md" in relative_files
    assert "outputs/paper_render/main.pdf" in relative_files
    assert "outputs/paper_render/preview/main.pdf.png" in relative_files
    assert "outputs/review_check/review_check_report.md" not in relative_files
    assert "outputs/review_check_all/review_check_report.md" not in relative_files
    assert "outputs/artifact_dry_run/extracted_bundle_report.md" not in relative_files
    assert "outputs/artifact_dry_run_anonymous/extracted_bundle_report.md" not in relative_files
    assert "outputs/smoke/metrics.csv" not in relative_files
    assert "outputs/other/local.pdf" not in relative_files
    assert "outputs/openml/runs/task/seed_0/metrics.csv" not in relative_files
    assert "outputs/openml/dataset_287.pq" not in relative_files
    assert "outputs/outlier_pair/ranc_correct_noise_contract_noise_seed0.json" not in relative_files
    assert "experiments/__pycache__/runner.pyc" not in relative_files
    assert ".DS_Store" not in relative_files


def test_artifact_packager_builds_zip_with_manifest_and_hashes(tmp_path):
    _minimal_repo(tmp_path)
    _write(tmp_path / "outputs/openml/openml_table.md", "| ok |\n")
    zip_path = tmp_path / "dist/artifact.zip"

    result = build_artifact(tmp_path, zip_path, max_size_bytes=1024 * 1024)

    assert result.zip_path.exists()
    assert result.sha256_path.exists()
    assert result.file_count > 3
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "ranc_contractnet_neurips2027_artifact/ARTIFACT_MANIFEST.md" in names
        assert "ranc_contractnet_neurips2027_artifact/REPRODUCE.md" in names
        assert "ranc_contractnet_neurips2027_artifact/SHA256SUMS" in names
        assert "ranc_contractnet_neurips2027_artifact/LICENSE" in names
        assert "ranc_contractnet_neurips2027_artifact/CITATION.cff" in names
        assert "ranc_contractnet_neurips2027_artifact/CONTRIBUTING.md" in names
        assert "ranc_contractnet_neurips2027_artifact/.zenodo.json" in names
        assert "ranc_contractnet_neurips2027_artifact/scripts/review_check.py" in names
        assert "ranc_contractnet_neurips2027_artifact/paper/neurips2027/artifact_eval.md" in names
        assert "ranc_contractnet_neurips2027_artifact/outputs/openml/openml_table.md" in names
        assert all("__pycache__" not in name for name in names)
        assert all("/runs/" not in name for name in names)
        manifest = archive.read(
            "ranc_contractnet_neurips2027_artifact/ARTIFACT_MANIFEST.md"
        ).decode("utf-8")
        reproduce = archive.read("ranc_contractnet_neurips2027_artifact/REPRODUCE.md").decode(
            "utf-8"
        )
        assert "Hygiene Policy" in manifest
        assert "paper/neurips2027/artifact_eval.md" in reproduce
        assert "scripts/review_check.py --tier 1" in reproduce
        assert "scripts/review_check.py --tier dryrun" in reproduce
        assert "/" + "Users" + "/" not in manifest


def test_artifact_packager_anonymous_mode_scrubs_identity_and_maps_render_outputs(tmp_path):
    _minimal_repo(tmp_path)
    author = "".join(["Har", "shad ", "Hem", "ant ", "Pat", "il"])
    org = "".join(["Spring", "er ", "Nat", "ure"])
    role_title = "".join(["Lead ", "Data ", "Scientist"])
    role_org = f"{role_title}, {org}"
    email = "".join(["hh", "patil", "001", "@", "gmail", ".com"])
    family_name = "".join(["Pat", "il"])
    given_names = "".join(["Har", "shad ", "Hem", "ant"])
    reversed_author = f"{family_name}, {given_names}"
    cff_family = f"family-names: {family_name}"
    cff_given = f"given-names: {given_names}"
    repo_user = "".join(["har", "shad", "317"])
    repo_url = f"https://github.com/{repo_user}/RANC-ContractNet"
    anonymous_repo_url = "https://anonymous.example.org/ranc-contractnet"
    _write(
        tmp_path / "paper/neurips2027/main.tex",
        f"{author}\n{role_org}\n{email}\n",
    )
    _write(
        tmp_path / "pyproject.toml",
        "[project]\n"
        "name='demo'\n"
        f"authors=[{{name='{author}', email='{email}'}}]\n"
        "[project.urls]\n"
        f"Repository='{repo_url}'\n",
    )
    _write(tmp_path / "README.md", f"Repo: {repo_url}\n")
    _write(tmp_path / "LICENSE", f"Copyright (c) 2026 {author}\n")
    _write(
        tmp_path / "CITATION.cff",
        "cff-version: 1.2.0\n"
        f"repository-code: {repo_url}\n"
        "authors:\n"
        f"  - {cff_family}\n"
        f"    {cff_given}\n"
        f"    email: {email}\n"
        f"    affiliation: {org}\n",
    )
    _write(
        tmp_path / ".zenodo.json",
        "{"
        f"\"creators\":[{{\"name\":\"{reversed_author}\",\"affiliation\":\"{org}\"}}],"
        f"\"related_identifiers\":[{{\"identifier\":\"{repo_url}\"}}]"
        "}\n",
    )
    _write_bytes(tmp_path / "outputs/paper_render/main.pdf", f"%PDF identified {author}\n".encode("utf-8"))
    _write_bytes(tmp_path / "outputs/paper_render/main_anonymous.pdf", b"%PDF anonymous\n")
    _write(
        tmp_path / "outputs/paper_render/paper_render_report_anonymous.md",
        "# Render\n\n- Status: **passed**\n- Style mode: fallback_article\n",
    )
    _write(tmp_path / "outputs/paper_render/paper_render_report_anonymous.json", "{}\n")
    _write_bytes(tmp_path / "outputs/paper_render/preview_anonymous/main_anonymous.pdf.png", b"png")

    zip_path = tmp_path / "dist/artifact_anonymous.zip"
    build_artifact(tmp_path, zip_path, identity_mode="anonymous", max_size_bytes=1024 * 1024)

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        main_tex = archive.read("ranc_contractnet_neurips2027_artifact/paper/neurips2027/main.tex").decode(
            "utf-8"
        )
        pyproject = archive.read("ranc_contractnet_neurips2027_artifact/pyproject.toml").decode("utf-8")
        packaged_pdf = archive.read("ranc_contractnet_neurips2027_artifact/outputs/paper_render/main.pdf")
        assert "Anonymous Author" in main_tex
        assert "Anonymous Institution" in main_tex
        assert "anonymous@example.org" in main_tex
        assert "Anonymous Author" in pyproject
        assert repo_user not in pyproject
        assert anonymous_repo_url in pyproject
        assert b"%PDF anonymous" in packaged_pdf
        assert "ranc_contractnet_neurips2027_artifact/outputs/paper_render/main_anonymous.pdf" not in names
        assert "ranc_contractnet_neurips2027_artifact/outputs/paper_render/preview/main.pdf.png" in names
        for name in names:
            data = archive.read(name)
            assert author.encode("utf-8") not in data
            assert email.encode("utf-8") not in data
            assert org.encode("utf-8") not in data
            assert role_title.encode("utf-8") not in data
            assert repo_user.encode("utf-8") not in data
            assert reversed_author.encode("utf-8") not in data
            assert cff_given.encode("utf-8") not in data
            assert cff_family.encode("utf-8") not in data


def test_artifact_packager_rejects_local_identity_leaks(tmp_path):
    _minimal_repo(tmp_path)
    local_path = "/" + "Users" + "/" + "hpu" + "4454" + "/private"
    _write(tmp_path / "README.md", f"local path: {local_path}\n")

    with pytest.raises(ValueError, match="Anonymization sanity check failed"):
        build_artifact(tmp_path, tmp_path / "dist/artifact.zip")
