# _gaussian_splats_sampled.py
#
# This file is intentionally a near-verbatim copy of _gaussian_splats.py.
# The ONLY functional differences from the original GS3D are:
#   1. the class name (GS3D_Sampled instead of GS3D), and
#   2. the shader `path=`, which points at decomposition_tracking_GS_sampled.h
#      instead of decomposition_tracking_GS.h.
# build_geometry_ads / build_ads (BLAS/TLAS construction, AABB sizing from the
# covariance) are unchanged on purpose: the acceleration structure is
# identical across all of GS3D, GS3D_Ratio, and GS3D_Sampled -- only the
# per-ray blending shader differs between them.
#
# Place this file next to _gaussian_splats.py inside the rdv package (i.e.
# wherever `from . import _core` resolves), and add one line to rdv's
# __init__.py:
#     from ._gaussian_splats_sampled import GS3D_Sampled
# so that `rdv.GS3D_Sampled` becomes available exactly like `rdv.GS3D`.

import torch as _torch
from . import _core
import vulky as _vk


class GS3D_SigmaT(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/decomposition_tracking_GS_sigma_t.h',
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

    # --- build_geometry_ads and build_ads: copied verbatim from _gaussian_splats.py ---
    # Nothing about acceleration-structure construction differs between the
    # delta- and ratio-tracking estimators, so this is a straight copy.

    def build_geometry_ads(self, vk_ads_info=None, just_update=False, reuse=True, **deferred) -> dict:
        """
        Builds the Bottom-Level Acceleration Structure (BLAS)
        """
        if isinstance(self.positions, _torch.Tensor):
            positions = self.positions
        else:
            positions = self.positions.evaluate(**deferred)

        if not reuse or vk_ads_info is None:
            aabb_buffer = _vk.structured_buffer(positions.shape[0], dict(
                bmin=_vk.vec3,
                bmax=_vk.vec3
            ), usage=_vk.BufferUsage.RAYTRACING_RESOURCE, memory=_vk.MemoryLocation.GPU)
        else:
            assert positions.shape[0] == vk_ads_info['num_primitives'], "Primitive count mismatch"
            aabb_buffer = vk_ads_info['aabb_buffer']

        # Ensure we have the raw covariance matrix (shape N, 6)
        if hasattr(self, 'covs') and self.covs is not None:
            # The diagonal elements of the 6-value covariance matrix are at indices 0, 3, and 5
            cov_xx = self.covs[:, 0]
            cov_yy = self.covs[:, 3]
            cov_zz = self.covs[:, 5]

            # The physical extent is 3 standard deviations (sqrt of variance)
            extents_x = (_torch.sqrt(cov_xx) * 3.0).unsqueeze(-1)
            extents_y = (_torch.sqrt(cov_yy) * 3.0).unsqueeze(-1)
            extents_z = (_torch.sqrt(cov_zz) * 3.0).unsqueeze(-1)

            # Stack them into a highly accurate 3D bounding box!
            extents = _torch.cat([extents_x, extents_y, extents_z], dim=-1)
        else:
            # Fallback to the static bounding volume with biggest gaussian
            max_scales = _torch.max(self.scales, dim=-1)[0]
            extents = (max_scales * 3.0).unsqueeze(-1)
            extents = extents.expand(-1, 3)  # Ensure it has 3 columns

        # Load the tight bounding boxes into the Vulkan hardware tree
        aabb_buffer.load(_torch.cat([positions - extents, positions + extents], dim=-1))

        if not reuse or vk_ads_info is None:
            vk_geometry = _vk.aabb_collection()
            vk_geometry.append(aabb=aabb_buffer)
            vk_geometry_ads = _vk.ads_model(vk_geometry)
            scratch_buffer = _vk.scratch_buffer(vk_geometry_ads)
        else:
            vk_geometry = vk_ads_info['vk_geometry']
            vk_geometry_ads = vk_ads_info['vk_geometry_ads']
            scratch_buffer = vk_ads_info['scratch_buffer']

        with _vk.raytracing_manager() as man:
            if not just_update or vk_ads_info is None:
                man.build_ads(vk_geometry_ads, scratch_buffer)
            else:
                man.update_ads(vk_geometry_ads, scratch_buffer)

        return dict(
            vk_geometry=vk_geometry,
            vk_geometry_ads=vk_geometry_ads,
            aabb_buffer=aabb_buffer,
            scratch_buffer=scratch_buffer,
            num_primitives=positions.shape[0],
            vk_geometry_ads_handle=vk_geometry_ads.handle
        )

    def build_ads(self, just_update=False, reuse=True, **deferred):
        """
        Builds the Top-Level Acceleration Structure (TLAS)
        """
        self.initialized = True
        self.vk_geometry_ads_info = self.build_geometry_ads(just_update=just_update, reuse=reuse, **deferred)
        transforms = _torch.zeros(1, 4, 3)
        transforms[0] = self.transform if not isinstance(self.transform, _core.deferred) else self.transform.evaluate(**deferred)

        if self.vk_ads_info is None or not reuse:
            geometry_ads_ptrs = _torch.tensor([[self.vk_geometry_ads_info['vk_geometry_ads_handle']]], dtype=_torch.int64)
            vk_instances = _vk.instance_buffer(1, memory=_vk.MemoryLocation.GPU)
            vk_scene_ads = _vk.ads_scene(vk_instances)
            scratch_buffer = _vk.scratch_buffer(vk_scene_ads)
        else:
            geometry_ads_ptrs = self.vk_ads_info['geometry_ads_ptrs']
            vk_instances = self.vk_ads_info['vk_instances']
            vk_scene_ads = self.vk_ads_info['vk_scene_ads']
            scratch_buffer = self.vk_ads_info['scratch_buffer']
            for i, geometry in enumerate(self.geometries):
                geometry.build_ads(just_update=just_update, reuse=True, **deferred)

        with vk_instances.map('in', clear=True) as s:
            s.flags = 0
            s.mask8_idx24 = _vk.asint32(0xFF000000)
            s.transform[0][0] = transforms[..., 0, 0]
            s.transform[1][0] = transforms[..., 0, 1]
            s.transform[2][0] = transforms[..., 0, 2]
            s.transform[0][1] = transforms[..., 1, 0]
            s.transform[1][1] = transforms[..., 1, 1]
            s.transform[2][1] = transforms[..., 1, 2]
            s.transform[0][2] = transforms[..., 2, 0]
            s.transform[1][2] = transforms[..., 2, 1]
            s.transform[2][2] = transforms[..., 2, 2]
            s.transform[0][3] = transforms[..., 3, 0]
            s.transform[1][3] = transforms[..., 3, 1]
            s.transform[2][3] = transforms[..., 3, 2]
            s.accelerationStructureReference = geometry_ads_ptrs

        with _vk.raytracing_manager() as man:
            if not just_update or self.vk_ads_info is None:
                man.build_ads(vk_scene_ads, scratch_buffer)
            else:
                man.update_ads(vk_scene_ads, scratch_buffer)

        self.vk_ads_info = dict(
            geometry_ads_ptrs=geometry_ads_ptrs,
            vk_instances=vk_instances,
            vk_scene_ads=vk_scene_ads,
            scratch_buffer=scratch_buffer
        )

        self.ads = _vk.wrap_gpu(vk_scene_ads.handle)
