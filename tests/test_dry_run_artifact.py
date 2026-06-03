import zipfile
from pathlib import Path

from scripts import dry_run_artifact


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_dry_run_checks_only_reports_packaged_render_artifacts(tmp_path):
    bundle = tmp_path / dry_run_artifact.BUNDLE_ROOT_NAME
    _write(bundle / "pyproject.toml", "[project]\nname='demo'\n")
    _write(bundle / "SHA256SUMS", "placeholder  pyproject.toml\n")
    _write(bundle / "outputs/paper_render/main.pdf", "%PDF generated render\n")
    _write(
        bundle / "outputs/paper_render/paper_render_report.md",
        "# Paper Render Report\n\n- Status: **passed**\n- Style mode: fallback_article\n",
    )
    _write(bundle / "outputs/paper_render/paper_render_report.json", "{}\n")
    _write(bundle / "outputs/paper_render/preview/main.pdf.png", "png preview\n")
    _write(bundle / "paper/neurips2027/STYLE_STATUS.md", "# Style Status\n")

    zip_path = tmp_path / "artifact.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(tmp_path).as_posix())
    zip_hash = dry_run_artifact.sha256_file(zip_path)
    zip_path.with_suffix(".sha256").write_text(f"{zip_hash}  {zip_path.name}\n", encoding="utf-8")

    report = dry_run_artifact.run_dry_run(
        zip_path=zip_path,
        output_dir=tmp_path / "reports",
        root=tmp_path,
        checks_only=True,
    )

    assert report["overall_status"] == "passed"
    assert report["commands"] == []
    assert report["zip_sha256"] == zip_hash
    assert (tmp_path / "reports/extracted_bundle_report.md").exists()
    assert (tmp_path / "reports/extracted_bundle_report.json").exists()
    text = (tmp_path / "reports/extracted_bundle_report.md").read_text(encoding="utf-8")
    assert "Extracted Bundle Artifact Dry Run" in text
    assert "intentionally not inserted into the artifact zip" in text
