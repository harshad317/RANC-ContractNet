"""Small CPU neural smoke runner for optional RANC activation policies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    try:
        import torch
        from torch import nn
    except Exception as exc:
        raise SystemExit(f"torch is not installed; install ranc-contractnet[torch] to run this: {exc}")

    from ranc_contractnet.torch_layers import RANCActivationPolicy

    torch.manual_seed(0)
    X = torch.randn(64, 8)
    y = (X[:, 0] - X[:, 1] > 0).long()
    layer = RANCActivationPolicy(
        {"hard_clauses": {"avoid_batch_dependence": True, "preserve_zero": False}},
        shape=8,
        random_state=0,
    )
    model = nn.Sequential(layer, nn.Linear(8, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(args.steps):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    output = Path("outputs/neural_smoke.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"loss={float(loss.detach().cpu()):.6f}\naudit={layer.last_audit}\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
