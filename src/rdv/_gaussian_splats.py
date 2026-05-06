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
            majorant_buffer=_torch.Tensor,
            minorant_buffer=_torch.Tensor,
            control_color_buffer =_torch.Tensor,
            grid_min=_torch.Tensor,
            grid_size=_torch.Tensor,
            scales = _torch.Tensor
        ),
        stochastic = True 
    )

    # NEW: Updated signature to accept inv_covs and opacities
    def __init__(self, positions: _core.TensorLike | _core.deferred, 
                 colors:_core.TensorLike | _core.deferred, 
                 inv_covs:_core.TensorLike | _core.deferred, 
                 opacities:_core.TensorLike | _core.deferred, 
                 transform: _core.TensorLike | _core.deferred = None, 
                 scales: _core.TensorLike | _core.deferred = None,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        
        positions = _core.ensure_tensor(positions, map_dim=2)
        colors = _core.ensure_tensor(colors, map_dim=2)
        inv_covs = _core.ensure_tensor(inv_covs, map_dim=2)
        opacities = _core.ensure_tensor(opacities, map_dim=1)
        
        assert len(positions.shape) == 2 and positions.shape[-1] == 3, "GS3D requires positions to be of shape (N, 3)"
        assert len(colors.shape) == 2, "GS3D requires colors to be of shape (N, C)"
        
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
        self.transform = transform
        self.scales = scales
        
        self.vk_geometry_ads_info = None
        self.vk_ads_info = None
        self.ads = None
        self.majorant_buffer = None
        self.minorant_buffer = None
        self.control_color_buffer = None
        self.grid_min = None
        self.grid_size = None

    def clone(self, **kwargs) -> 'Map':
        return GS3D(
            positions=self.positions,
            colors=self.colors,
            inv_covs=self.inv_covs,
            opacities=self.opacities,
            transform=self.transform,
            **kwargs
        )

    
    def _build_decomposition_grid(self, grid_resolution=64):
        """Helper function to calculate both Majorant and Minorant grids"""
        
        # Majorant Calculation
        voxel_grid = _torch.full((grid_resolution, grid_resolution, grid_resolution), 
                                fill_value=0.1, dtype=_torch.float32, device=self.positions.device)

        min_bounds = self.positions.min(dim=0)[0]
        max_bounds = self.positions.max(dim=0)[0]
        scene_size = max_bounds - min_bounds
        
        min_bounds -= scene_size * 0.01
        max_bounds += scene_size * 0.01
        scene_size = max_bounds - min_bounds

        norm_positions = (self.positions - min_bounds) / scene_size
        grid_indices = (norm_positions * (grid_resolution - 1)).long()

        idx_x = grid_indices[:, 0]
        idx_y = grid_indices[:, 1]
        idx_z = grid_indices[:, 2]

        peak_densities = self.opacities.squeeze() * 5.0

        flat_indices = idx_x * (grid_resolution ** 2) + idx_y * grid_resolution + idx_z
        flat_grid = voxel_grid.flatten()
        flat_grid.scatter_reduce_(dim=0, index=flat_indices, src=peak_densities, reduce='amax', include_self=True)
        
        self.majorant_buffer = flat_grid.contiguous()
        self.grid_min = min_bounds.contiguous()
        self.grid_size = scene_size.contiguous()
       

        # Calculate Minorant Grid
        # Improved Minorant Logic
        # Initialize with a very high value so amin can actually find the lowest peak_density
        voxel_min_grid = _torch.full((grid_resolution**3,), 1000.0, device=self.positions.device)

        # Only scatter to voxels that actually have Gaussians
        flat_min_grid = voxel_min_grid.scatter_reduce_(
            dim=0, 
            index=flat_indices, 
            src=peak_densities, 
            reduce='amin', 
            include_self=False # Don't include the 1000.0 initial value
        )

        # Fill voxels that never received a value with 0.0
        flat_min_grid[flat_min_grid == 1000.0] = 0.0

        # Apply a conservative multiplier (e.g., 0.1) 
        # The paper notes that underestimating is safe but overestimating causes bias
        self.minorant_buffer = (flat_min_grid * 0.1).contiguous()


        # Initialize a grid for RGB sums and a counter grid
        color_sum_grid = _torch.zeros((grid_resolution**3, 3), device=self.positions.device)
        count_grid = _torch.zeros((grid_resolution**3,), device=self.positions.device)

        # Use scatter_add to sum up colors and counts per voxel
        # colors shape: (N, 3), flat_indices shape: (N)
        color_sum_grid.scatter_add_(0, flat_indices.unsqueeze(-1).expand(-1, 3), self.colors)
        count_grid.scatter_add_(0, flat_indices, _torch.ones_like(flat_indices, dtype=_torch.float32))

        # Calculate average: Sum / Count (avoid division by zero)
        safe_count = count_grid.unsqueeze(-1).clamp(min=1.0)
        mean_color_grid = color_sum_grid / safe_count

        self.control_color_buffer = mean_color_grid.contiguous()

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

        max_scales = _torch.max(self.scales, dim=-1)[0]
        
        # A Gaussian physically ends at roughly 3.0 standard deviations
        extents = (max_scales * 3.0).unsqueeze(-1) 
        
        # Load perfectly tight bounding boxes into the Vulkan hardware tree!
        aabb_buffer.load(_torch.cat([positions - extents, positions + extents], dim=-1))

        # aabb_buffer.load(_torch.cat([positions - self.radius, positions + self.radius], dim=-1))
        
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
        Builds the Top-Level Acceleration Structure (TLAS) and the Voxel Grid
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
        
        # Voxel Grid after Ray Tracing ADS
        self._build_decomposition_grid(grid_resolution=64)