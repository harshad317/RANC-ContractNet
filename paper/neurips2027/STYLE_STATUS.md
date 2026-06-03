# NeurIPS Style Status

## Current Status

The current repository does not include `paper/neurips2027/neurips_2027.sty`.
Therefore `paper/neurips2027/main.tex` renders through its fallback `article`
path with 1-inch margins.

The current PDF render is a clean source-level and fallback-layout check. It is
not a claim of official NeurIPS 2027 style compliance.

## Style Switch

`main.tex` uses:

```tex
\IfFileExists{neurips_2027.sty}{
  \usepackage[preprint]{neurips_2027}
}{
  \usepackage[margin=1in]{geometry}
}
```

When the official style file is available, place it at:

```text
paper/neurips2027/neurips_2027.sty
```

Then rerun:

```bash
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier render
PATH="/Library/TeX/texbin:$PATH" python3 scripts/review_check.py --tier render_anonymous
python3 -m pytest
python3 scripts/review_check.py --tier package
python3 scripts/review_check.py --tier package_anonymous
```

## Official-Style Checklist

- Confirm the render report says `Style mode: official_neurips_2027`.
- Confirm the anonymous render and anonymous bundle still pass before submission.
- Recheck page count under the official style.
- Recheck all imported result tables for overfull boxes, clipping, or unreadable
  text.
- Recheck citations and references for unresolved keys.
- Confirm the checklist, supplementary material, artifact guide, rendered PDF,
  and SHA256 bundle are packaged.
- Update this file with the official-style render date and page count.
