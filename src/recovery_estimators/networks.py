"""The one network both estimators use, so a comparison is about the objective.

Derived from identifiability-research-1, `src/idiff/score.py` (conditional score MLP with a
seeded Gaussian Fourier time embedding). Modified: the checkpoint reuse, training snapshots,
best-state selection and loss history of that file are removed; the time embedding is optional
so the same trunk serves a time-free classifier; the output width is a parameter.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import torch
from torch import nn

from recovery_core.types import Array


@dataclass(frozen=True)
class NetworkConfig:
    width: int = 512
    hidden_layers: int = 4
    embedding_dim: int = 64
    frequency_scale: float = 16.
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if (min(self.width, self.hidden_layers) < 1 or self.embedding_dim % 2
                or self.frequency_scale <= 0 or self.dtype not in ("float32", "float64")):
            raise ValueError("Invalid network configuration")


class ConditionMLP(nn.Module):
    """concat(x, one_hot(u) if conditioned, time embedding if timed) -> outputs."""

    frequencies: torch.Tensor

    def __init__(self, d: int, n_conditions: int, outputs: int, config: NetworkConfig,
                 seed: int, *, timed: bool, conditioned: bool) -> None:
        super().__init__()
        self.n_conditions, self.timed, self.conditioned = n_conditions, timed, conditioned
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.register_buffer("frequencies",
                                 torch.randn(config.embedding_dim//2)*config.frequency_scale)
            incoming = d + (n_conditions if conditioned else 0) \
                + (config.embedding_dim if timed else 0)
            layers: list[nn.Module] = []
            for _ in range(config.hidden_layers):
                layers += [nn.Linear(incoming, config.width), nn.SiLU()]
                incoming = config.width
            layers.append(nn.Linear(incoming, outputs))
            self.network = nn.Sequential(*layers)
        self.to(device=config.device, dtype=getattr(torch, config.dtype))

    def forward(self, x: torch.Tensor, u: torch.Tensor | None = None,
                t: torch.Tensor | None = None) -> torch.Tensor:
        parts = [x]
        if self.timed:
            if t is None:
                raise ValueError("A timed network needs t")
            phase = 2*torch.pi*t[:, None]*self.frequencies[None]
            parts.append(torch.cat((torch.sin(phase), torch.cos(phase)), dim=1))
        if self.conditioned:
            if u is None:
                raise ValueError("A conditioned network needs u")
            parts.append(torch.nn.functional.one_hot(u, self.n_conditions).to(x.dtype))
        return self.network(torch.cat(parts, dim=1))


def state_hash(model: nn.Module) -> str:
    digest = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def batches(n: int, size: int, updates: int, seed: int) -> Any:
    generator = torch.Generator().manual_seed(seed)
    for _ in range(updates):
        yield torch.randint(n, (size,), generator=generator)


def as_tensor(x: Array, config: NetworkConfig) -> torch.Tensor:
    return torch.as_tensor(x, dtype=getattr(torch, config.dtype), device=config.device)
