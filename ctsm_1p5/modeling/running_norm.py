"""Per-patch updates of masked running mean and standard deviation."""

import torch


def update_running_stats(
    n: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool the current patch into the previous population statistics."""
    observed = torch.logical_not(mask)
    patch_count = observed.sum(dim=-1, dtype=x.dtype)
    empty_patch = patch_count == 0
    patch_divisor = torch.where(empty_patch, 1, patch_count)

    patch_mean = (x * observed).sum(dim=-1) / patch_divisor
    patch_mean = torch.where(empty_patch, 0, patch_mean)
    patch_offsets = x - patch_mean[:, None]
    patch_variance = (patch_offsets.square() * observed).sum(dim=-1) / patch_divisor
    patch_variance = torch.where(empty_patch, 0, patch_variance)

    count = n + patch_count
    empty = count == 0
    divisor = torch.where(empty, 1, count)
    mean = (n * mu + patch_mean * patch_count) / divisor
    mean = torch.where(empty, 0, mean)

    previous_mass = n * sigma.square()
    patch_mass = patch_count * patch_variance
    previous_shift = n * (mu - mean).square()
    patch_shift = patch_count * (patch_mean - mean).square()
    variance = (previous_mass + patch_mass + previous_shift + patch_shift) / divisor
    variance = torch.where(empty, 0, variance)

    variance = variance.clamp_min(0)
    positive = variance > 0
    safe_variance = torch.where(positive, variance, 1)
    deviation = torch.where(positive, safe_variance.sqrt(), 0)

    return count, mean, deviation
