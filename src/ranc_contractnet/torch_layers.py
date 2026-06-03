"""Optional torch activation policies for RANC-ContractNet."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:  # pragma: no cover - covered by optional dependency environments
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from ranc_contractnet.schemas import InvarianceContract


BaseModule = nn.Module if nn is not None else object


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("torch is required for ranc_contractnet.torch_layers. Install with .[torch].")


def _contract_hard(contract: object, name: str, default: bool = False) -> bool:
    if isinstance(contract, InvarianceContract):
        return contract.hard(name, default)
    if isinstance(contract, dict):
        return bool(contract.get("hard_clauses", {}).get(name, default))
    return default


class RANCActivationPolicy(BaseModule):
    """Contract-driven activation normalizer.

    The layer is intentionally lightweight: it chooses centering/no-centering, axes,
    batch dependence, and bounded nonlinear guards from explicit contract clauses.
    """

    def __init__(
        self,
        contract: Optional[object],
        shape: Sequence[int] | int,
        random_state: int = 0,
        eps: float = 1e-5,
    ) -> None:
        _require_torch()
        super().__init__()
        if isinstance(shape, int):
            normalized_shape = (shape,)
        else:
            normalized_shape = tuple(int(v) for v in shape)
        self.contract = contract or {"hard_clauses": {}}
        self.normalized_shape = normalized_shape
        self.random_state = int(random_state)
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.last_audit: Dict[str, Any] = {}

    def forward(self, x):  # type: ignore[override]
        _require_torch()
        avoid_batch = _contract_hard(self.contract, "avoid_batch_dependence", True)
        forbid_centering = _contract_hard(self.contract, "forbid_centering", False) or _contract_hard(
            self.contract, "preserve_zero", False
        )
        use_bounded = _contract_hard(self.contract, "bounded_nonlinearity", False)
        dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        if not avoid_batch:
            dims = tuple(range(x.ndim))
        if forbid_centering:
            denom = torch.sqrt(torch.mean(x * x, dim=dims, keepdim=True) + self.eps)
            z = x / denom
            centering = "rms"
        else:
            mean = torch.mean(x, dim=dims, keepdim=True)
            var = torch.mean((x - mean) ** 2, dim=dims, keepdim=True)
            z = (x - mean) / torch.sqrt(var + self.eps)
            centering = "mean"
        if use_bounded:
            z = torch.tanh(z)
            saturation = torch.mean((torch.abs(z) > 0.98).float()).detach()
            saturation_rate = float(saturation.cpu().item())
        else:
            saturation_rate = 0.0
        self.last_audit = {
            "avoid_batch_dependence": avoid_batch,
            "centering": centering,
            "bounded_nonlinearity": use_bounded,
            "saturation_rate": saturation_rate,
            "training": bool(self.training),
        }
        return z * self.gamma + self.beta


class RANCNormAdapter(BaseModule):
    """Thin adapter around common normalization baselines for controlled comparisons."""

    def __init__(self, name: str, shape: Sequence[int] | int, **kwargs: Any) -> None:
        _require_torch()
        super().__init__()
        key = name.lower()
        if isinstance(shape, int):
            normalized_shape = shape
        else:
            normalized_shape = tuple(int(v) for v in shape)
        if key == "layernorm":
            self.layer = nn.LayerNorm(normalized_shape, **kwargs)
        elif key == "batchnorm1d":
            features = normalized_shape if isinstance(normalized_shape, int) else int(normalized_shape[-1])
            self.layer = nn.BatchNorm1d(features, **kwargs)
        elif key == "groupnorm":
            features = normalized_shape if isinstance(normalized_shape, int) else int(normalized_shape[-1])
            groups = int(kwargs.pop("num_groups", 1))
            self.layer = nn.GroupNorm(groups, features, **kwargs)
        elif key == "rmsnorm":
            self.layer = _RMSNorm(normalized_shape, **kwargs)
        elif key == "dyt":
            self.layer = _DyT(normalized_shape, **kwargs)
        else:
            raise ValueError(f"Unknown adapter {name!r}.")

    def forward(self, x):  # type: ignore[override]
        return self.layer(x)


class _RMSNorm(BaseModule):
    def __init__(self, shape: Sequence[int] | int, eps: float = 1e-5) -> None:
        _require_torch()
        super().__init__()
        normalized_shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps

    def forward(self, x):  # type: ignore[override]
        dims = tuple(range(x.ndim - self.weight.ndim, x.ndim))
        rms = torch.sqrt(torch.mean(x * x, dim=dims, keepdim=True) + self.eps)
        return x / rms * self.weight


class _DyT(BaseModule):
    def __init__(self, shape: Sequence[int] | int, alpha: float = 1.0) -> None:
        _require_torch()
        super().__init__()
        normalized_shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):  # type: ignore[override]
        return torch.tanh(self.alpha * x) * self.gamma + self.beta


def make_norm_adapter(name: str, shape: Sequence[int] | int, **kwargs: Any) -> RANCNormAdapter:
    return RANCNormAdapter(name, shape, **kwargs)

