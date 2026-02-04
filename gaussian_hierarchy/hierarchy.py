from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from . import _C


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    scales: Float[Tensor, "batch gaussian 3"]
    rotations: Float[Tensor, "batch gaussian 4"]


def _rotation_matrix_from_quaternion(rotations: Tensor) -> Tensor:
    s = rotations[..., 0]
    x = rotations[..., 1]
    y = rotations[..., 2]
    z = rotations[..., 3]

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - s * z)
    r02 = 2.0 * (x * z + s * y)

    r10 = 2.0 * (x * y + s * z)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - s * x)

    r20 = 2.0 * (x * z - s * y)
    r21 = 2.0 * (y * z + s * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def _compute_covariances(scales: Tensor, rotations: Tensor) -> Tensor:
    rotation_matrices = _rotation_matrix_from_quaternion(rotations)
    l = torch.diag_embed(scales)
    t = rotation_matrices @ l
    return t @ t.transpose(-1, -2)


def _indices_for_depth(nodes: Tensor, depth: int) -> Tensor:
    depth_mask = nodes[:, 0] == depth
    starts = nodes[depth_mask, 2].tolist()
    leaf_counts = nodes[depth_mask, 3].tolist()
    merged_counts = nodes[depth_mask, 4].tolist()

    ranges = []
    for start, leaf_count, merged_count in zip(starts, leaf_counts, merged_counts):
        count = int(leaf_count + merged_count)
        if count == 0:
            continue
        ranges.append(torch.arange(start, start + count, dtype=torch.long))

    if not ranges:
        return torch.empty((0,), dtype=torch.long)

    return torch.cat(ranges, dim=0)


def _indices_for_level(nodes: Tensor, num_gaussian: int) -> Tensor:
    if num_gaussian <= 0:
        raise ValueError("num_gaussian must be a positive integer.")

    depths = torch.unique(nodes[:, 0]).tolist()
    for depth in sorted(depths):
        indices = _indices_for_depth(nodes, depth)
        if indices.numel() >= num_gaussian:
            return indices[:num_gaussian]

    raise ValueError("num_gaussian exceeds the available Gaussians at any level.")


def gaussian_hierarchy_subsampling(gaussians: Gaussians, num_gaussian: int) -> Gaussians:
    (
        positions,
        harmonics,
        opacities,
        log_scales,
        rotations,
        _nodes,
        _boxes) = _C.build_hierarchy(
        gaussians.means,
        gaussians.harmonics,
        gaussians.opacities,
        gaussians.scales,
        gaussians.rotations,
    )

    scales = log_scales.exp()
    covariances = _compute_covariances(scales, rotations)
    indices = _indices_for_level(_nodes, num_gaussian)

    return Gaussians(
        means=positions[indices],
        covariances=covariances[indices],
        harmonics=harmonics[indices],
        opacities=opacities.squeeze(-1)[indices],
        scales=scales[indices],
        rotations=rotations[indices],
    )
