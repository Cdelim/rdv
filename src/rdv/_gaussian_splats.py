import torch as _torch
from . import _core
import vulky as _vk

import torch as _torch
from . import _core
import vulky as _vk

class GS3D(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/decomposition_tracking_GS.h',
        parameters=dict(
            ads=_torch.int64,
            positions=_torch.Tensor,
            colors=_torch.Tensor,
            inv_covs=_torch.Tensor,
            opacities=_torch.Tensor,
            scales = _torch.Tensor,
            f_rest = _torch.Tensor, 
            covs = _torch.Tensor  
        ),
        stochastic = True  # Keep this so your C++ random() works!
    )

    def __init__(self, positions: _core.TensorLike | _core.deferred, 
                 colors:_core.TensorLike | _core.deferred, 
                 inv_covs:_core.TensorLike | _core.deferred,
                 covs:_core.TensorLike | _core.deferred,   
                 opacities:_core.TensorLike | _core.deferred, 
                 f_rest:_core.TensorLike | _core.deferred, 
                 transform: _core.TensorLike | _core.deferred = None, 
                 scales: _core.TensorLike | _core.deferred = None,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        
        positions = _core.ensure_tensor(positions, map_dim=2)
        colors = _core.ensure_tensor(colors, map_dim=2)
        inv_covs = _core.ensure_tensor(inv_covs, map_dim=2)
        covs = _core.ensure_tensor(covs, map_dim=2)
        opacities = _core.ensure_tensor(opacities, map_dim=1)
        f_rest = _core.ensure_tensor(f_rest, map_dim=2) # 3. ADDED: Ensure it is a valid 2D tensor
        
        assert len(positions.shape) == 2 and positions.shape[-1] == 3, "GS3D requires positions to be of shape (N, 3)"
        assert len(colors.shape) == 2, "GS3D requires colors to be of shape (N, C)"
        assert f_rest.shape[-1] == 45, f"GS3D requires f_rest to have 45 columns, got {f_rest.shape[-1]}" # 4. ADDED: Safety check
        
        if output_dim is None:
            output_dim = colors.shape[-1]  
        assert output_dim == colors.shape[-1], f"GS3D output_dim must match, got {output_dim} but colors has {colors.shape}"
        
        if transform is None:
            transform = _vk.mat4x3.trs()
        transform = _core.ensure_tensor(transform, map_dim=2)
        
        if input_dim is None:
            input_dim = 6
        assert input_dim == 6, f"GS3D only supports input_dim=6 (x, w), got input_dim={input_dim}"
        
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output
        )
        
        self.positions = positions
        self.colors = colors
        self.inv_covs = inv_covs
        self.opacities = opacities
        self.f_rest = f_rest      # 5. ADDED: Save it to the instance
        self.transform = transform
        self.scales = scales
        self.covs = covs
        
        self.vk_geometry_ads_info = None
        self.vk_ads_info = None
        self.ads = None

    def clone(self, **kwargs) -> 'Map':
        return GS3D(
            positions=self.positions,
            colors=self.colors,
            inv_covs=self.inv_covs,
            opacities=self.opacities,
            f_rest=self.f_rest,   # 6. ADDED: Ensure clone passes it down!
            transform=self.transform,
            scales=self.scales,
            covs=self.covs,
            **kwargs
        )

    # ... (Keep build_geometry_ads and build_ads exactly the same!) ...

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
            extents = extents.expand(-1, 3) # Ensure it has 3 columns

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
        
        # We no longer call _build_decomposition_grid here!