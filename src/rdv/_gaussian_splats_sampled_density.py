# condor_data_loader.py
#
# Loads the "Don't Splat your Gaussians" paper's own .ply schema (different
# from standard 3DGS) and wires it up to a new shader that uses the raw
# density directly instead of going through 3DGS's opacity approximation.
#
# Paste into new cells in your notebook. Needs plyfile (you already
# `pip install`ed it earlier), torch, rdv, and compute_covariance /
# compute_inverse_covariance already defined.

import torch as _torch
from . import _core
import vulky as _vk


class GS3D_SigmaT(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/decomposition_tracking_GS_sampled_withDensity.h',
        parameters=dict(
            ads=_torch.int64,
            positions=_torch.Tensor,
            colors=_torch.Tensor,
            inv_covs=_torch.Tensor,
            sigma_t=_torch.Tensor,     # <-- the actual difference from GS3D_Sampled
            scales=_torch.Tensor,
            f_rest=_torch.Tensor,
            covs=_torch.Tensor
        ),
        stochastic=True,
    )

    def __init__(self, positions, colors, inv_covs, covs, sigma_t, f_rest,
                 transform=None, scales=None,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):

        positions = _core.ensure_tensor(positions, map_dim=2)
        colors = _core.ensure_tensor(colors, map_dim=2)
        inv_covs = _core.ensure_tensor(inv_covs, map_dim=2)
        covs = _core.ensure_tensor(covs, map_dim=2)
        sigma_t = _core.ensure_tensor(sigma_t, map_dim=1)
        f_rest = _core.ensure_tensor(f_rest, map_dim=2)

        assert positions.shape[-1] == 3
        assert f_rest.shape[-1] == 45, f"expected 45 f_rest columns, got {f_rest.shape[-1]}"

        if output_dim is None:
            output_dim = colors.shape[-1]
        if transform is None:
            transform = _vk.mat4x3.trs()
        transform = _core.ensure_tensor(transform, map_dim=2)
        if input_dim is None:
            input_dim = 6

        super().__init__(input_dim=input_dim, output_dim=output_dim,
                          input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)

        self.positions = positions
        self.colors = colors
        self.inv_covs = inv_covs
        self.sigma_t = sigma_t
        self.f_rest = f_rest
        self.transform = transform
        self.scales = scales
        self.covs = covs
        self.vk_geometry_ads_info = None
        self.vk_ads_info = None
        self.ads = None

    def clone(self, **kwargs):
        return GS3D_SigmaT(
            positions=self.positions, colors=self.colors, inv_covs=self.inv_covs,
            covs=self.covs, sigma_t=self.sigma_t, f_rest=self.f_rest,
            transform=self.transform, scales=self.scales, **kwargs
        )

    # build_geometry_ads / build_ads: copy verbatim from _gaussian_splats.py
    # (identical BLAS/TLAS logic -- see that file, or _gaussian_splats_ratio.py
    # for the full text). Omitted here to avoid a fourth copy of the same
    # ~90 lines; go get it from either of those two files.

