from collections.abc import Callable

import torch


TimeSampler = Callable[[int, torch.device | str, torch.dtype], torch.Tensor]


def sample_uniform_time(
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample t uniformly from [0, 1], returning shape [B, 1]."""

    return torch.rand(batch_size, 1, device=device, dtype=dtype)


def _sample_symmetric_beta_time(
    batch_size: int,
    concentration: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.tensor(concentration, device=device, dtype=dtype)
    distribution = torch.distributions.Beta(value, value)
    t = distribution.sample((batch_size,)).reshape(batch_size, 1)
    eps = torch.finfo(dtype).eps
    return t.clamp(min=eps, max=1.0 - eps)


def sample_center_time(
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample more often near t=0.5 with a symmetric Beta(2, 2)."""

    return _sample_symmetric_beta_time(batch_size, 2.0, device, dtype)


def sample_endpoint_time(
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample more often near t=0 and t=1 with a Beta(0.5, 0.5)."""

    return _sample_symmetric_beta_time(batch_size, 0.5, device, dtype)


TIME_SAMPLERS: dict[str, TimeSampler] = {
    "uniform": sample_uniform_time,
    "center": sample_center_time,
    "endpoint": sample_endpoint_time,
}


def get_time_sampler(name: str) -> TimeSampler:
    """Return a named sampler used by the teaching scripts."""

    try:
        return TIME_SAMPLERS[name]
    except KeyError as exc:
        choices = ", ".join(TIME_SAMPLERS)
        raise ValueError(f"unknown time sampler {name!r}; choose from: {choices}") from exc
