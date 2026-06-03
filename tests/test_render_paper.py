from pathlib import Path

from scripts import render_paper


def test_render_paper_skips_without_tex_engine(tmp_path, monkeypatch):
    paper_dir = tmp_path / "paper" / "neurips2027"
    paper_dir.mkdir(parents=True)
    (paper_dir / "main.tex").write_text("\\documentclass{article}\\begin{document}x\\end{document}\n", encoding="utf-8")

    monkeypatch.setattr(render_paper.shutil, "which", lambda _name: None)

    report = render_paper.run_render_check(
        root=tmp_path,
        output_dir=tmp_path / "outputs" / "paper_render",
        timeout_seconds=1,
    )

    assert report["status"] == "skipped"
    assert report["style_mode"] == "fallback_article"
    assert report["style_file_present"] is False
    assert "No TeX engine found" in report["reason"]
    assert (tmp_path / "outputs" / "paper_render" / "paper_render_report.md").exists()
    assert (tmp_path / "outputs" / "paper_render" / "paper_render_report.json").exists()


def test_render_paper_anonymous_variant_uses_wrapper_and_variant_reports(tmp_path, monkeypatch):
    paper_dir = tmp_path / "paper" / "neurips2027"
    paper_dir.mkdir(parents=True)
    (paper_dir / "main.tex").write_text("\\documentclass{article}\\begin{document}x\\end{document}\n", encoding="utf-8")
    (paper_dir / "main_anonymous.tex").write_text("\\def\\rancanonymoussubmission{1}\\input{main.tex}\n", encoding="utf-8")

    monkeypatch.setattr(render_paper.shutil, "which", lambda _name: None)

    report = render_paper.run_render_check(
        root=tmp_path,
        output_dir=tmp_path / "outputs" / "paper_render",
        timeout_seconds=1,
        variant="anonymous",
    )

    assert report["status"] == "skipped"
    assert report["variant"] == "anonymous"
    assert report["source_tex"] == "main_anonymous.tex"
    assert (tmp_path / "outputs" / "paper_render" / "paper_render_report_anonymous.md").exists()
    assert (tmp_path / "outputs" / "paper_render" / "paper_render_report_anonymous.json").exists()
