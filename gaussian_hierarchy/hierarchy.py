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


def gaussian_hierarchy_subsampling(gaussians: Gaussians) -> Gaussians:
    (
        positions,
        harmonics,
        opacities,
        log_scales,
        rotations,
        _nodes,
        _boxes,
    ) = _C.build_hierarchy(
        gaussians.means,
        gaussians.harmonics,
        gaussians.opacities,
        gaussians.scales,
        gaussians.rotations,
    )

    scales = log_scales.exp()
    covariances = _compute_covariances(scales, rotations)

    return Gaussians(
        means=positions,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities.squeeze(-1),
        scales=scales,
        rotations=rotations,
    )
