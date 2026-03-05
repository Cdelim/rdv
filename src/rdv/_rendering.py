import typing
from functools import cached_property

import mpmath
import torch
import torch as _torch
import typing as _typing

import vulky as _vk
from . import _core
import numpy as _np


# ========== Sensor Probes Map ==========

class CameraProbes(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/perspective_camera_sensors.h",
        parameters=dict(
            camera_poses=_torch.Tensor,
            num_cameras=int,
            fov=float,
            aspect_ratio=float,
        )
    )

    def __init__(self,
                 camera_poses: _core.TensorLike | _core.deferred,
                 fov: float = _core.pi / 4,
                 aspect_ratio: float = 1.0,
                 input_dim=None, output_dim=None, input_requires_grad=None, bw_uses_output=None
                 ):
        if input_dim is None:
            input_dim = 3  # x,y screen space coord, z (camera index)
        if output_dim is None:
            output_dim = 6  # ray origin (3) + ray direction (3)
        assert input_dim == 3, "CameraProbes requires input_dim of 3: x,y screen space coord, z (camera index)"
        assert output_dim == 6, "CameraProbes requires output_dim of 6 for rays: origin (3) + direction (3)"
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        camera_poses = _core.ensure_tensor(camera_poses, 2)
        assert camera_poses.shape[1] == 9, "Camera poses must be of shape (N, 9) for position, target and up"
        self.camera_poses = camera_poses
        self.num_cameras = camera_poses.shape[0]
        self.fov = fov
        self.aspect_ratio = aspect_ratio

    def clone(self,
              **kwargs) -> _core.Map:
        return CameraProbes(
            self.camera_poses,
            self.fov,
            self.aspect_ratio,
            **kwargs)


# =========== Scene Description =============

class Geometry:
    def patch_info(self, **deferred_parameters) -> dict:
        """
        Returns a dictionary with information about the geometry patch.
        Deferred parameters must be used to evaluate deferred tensors.
        'positions': Tensor of shape (N, 3) with the vertex positions.
        'indices': Tensor of shape (M, 3) with the triangle indices.
        """
        raise NotImplementedError()

    def mesh_info(self):
        """
        Returns a dictionary with all mesh information needed for raycasting.
        Tensors could be deferred tensors.
        'positions': Tensor of shape (N, 3) with the vertex positions.
        'indices': Tensor of shape (M, 3) with the triangle indices.
        'normals': Tensor of shape (N, 3) with the vertex normals.
        'uvs': Tensor of shape (N, 2) with the vertex UV coordinates.
        """
        raise NotImplementedError()

    def surface_map(self, transform: _vk.mat4x3 | _core.TensorLike | _core.deferred = _vk.mat4x3.trs()) -> _core.Map:
        """
        Returns a SurfacesMap for this geometry with the given transform.
        """
        return SurfacesMap(geometries=[self], transforms=[transform])

    def surfel_map(self,
                    transform: _vk.mat4x3 | _core.TensorLike | _core.deferred = _vk.mat4x3.trs(),
                    surfels_parameters: str = 'GNC') -> _core.Map:
        """
        Returns a SurfelsMap for this geometry with the given transform and surfel parameters.
        surfels_parameters: string containing the surfel components to include.
            'P': position
            'N': shading normal
            'G': geometry normal
            'C': uv coordinates
        """
        return SurfelsMap(geometries=[self], transforms=[transform], surfels_parameters=surfels_parameters)


class Box(Geometry):
    def __init__(self, as_mesh=False):
        self._patch_info = dict(
            positions=_vk.tensor_from(
                [
                    # Neg z
                    [-1.0, -1.0, -1.0],  # 0
                    [1.0, -1.0, -1.0],  # 1
                    [-1.0, 1.0, -1.0],  # 2
                    [1.0, 1.0, -1.0],  # 3
                    # Pos z

                    [-1.0, -1.0, 1.0],  # 4
                    [1.0, -1.0, 1.0],  # 5
                    [-1.0, 1.0, 1.0],  # 6
                    [1.0, 1.0, 1.0],  # 7

                    [-1.0, -1.0, -1.0],  # 8
                    [1.0, -1.0, -1.0],  # 9
                    [-1.0, -1.0, 1.0],  # 10
                    [1.0, -1.0, 1.0],  # 11

                    [-1.0, 1.0, -1.0],  # 12
                    [1.0, 1.0, -1.0],  # 13
                    [-1.0, 1.0, 1.0],  # 14
                    [1.0, 1.0, 1.0],  # 15

                    [-1.0, -1.0, -1.0],  # 16
                    [-1.0, 1.0, -1.0],  # 17
                    [-1.0, -1.0, 1.0],  # 18
                    [-1.0, 1.0, 1.0],  # 19

                    [1.0, -1.0, -1.0],  # 20
                    [1.0, 1.0, -1.0],  # 21
                    [1.0, -1.0, 1.0],  # 22
                    [1.0, 1.0, 1.0],  # 23
                ]
            ),
            indices=_vk.tensor_from(
                [
                    [0, 2, 3],
                    [0, 3, 1],
                    [4, 7, 6],
                    [4, 5, 7],

                    [8, 11, 10],
                    [8, 9, 11],
                    [12, 14, 15],
                    [12, 15, 13],

                    [16, 18, 19],
                    [16, 19, 17],
                    [20, 23, 22],
                    [20, 21, 23],
                ], dtype=_torch.int32
            ), normals=None, uvs=_vk.tensor_from([
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],

                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ])
        )
        self.as_mesh = as_mesh

    def mesh_info(self):
        return self._patch_info

    def patch_info(self, **deferred_parameters) -> dict:
        return self._patch_info

    def surface_map(self, transform: _vk.mat4x3 | _core.TensorLike | _core.deferred = _vk.mat4x3.trs()) -> _core.Map:
        if self.as_mesh:
            return super().surface_map(transform)
        return BoxSurfaceMap(transform)

    def surfel_map(self,
                    transform: _vk.mat4x3 | _core.TensorLike | _core.deferred = _vk.mat4x3.trs(),
                    surfels_parameters: str = 'GNC') -> _core.Map:
        if self.as_mesh:
            return super().surfel_map(transform, surfels_parameters)
        return BoxSurfelMap(transform, surfels_parameters)


class Mesh(Geometry):

    @staticmethod
    def compute_normals(positions: _torch.Tensor, indices: _torch.Tensor):
        normals = _torch.zeros_like(positions)
        indices = indices.long()  # to be used
        P0 = positions[indices[:, 0]]
        P1 = positions[indices[:, 1]]
        P2 = positions[indices[:, 2]]
        V1 = _vk.vec3.normalize(P1 - P0)
        V2 = _vk.vec3.normalize(P2 - P0)
        N = _torch.cross(V1, V2)  # do not normalize for proper weight
        indices0 = indices[:, 0].unsqueeze(-1).repeat(1, 3)
        normals.scatter_add_(0, indices0, N)
        indices1 = indices[:, 1].unsqueeze(-1).repeat(1, 3)
        normals.scatter_add_(0, indices1, N)
        indices2 = indices[:, 2].unsqueeze(-1).repeat(1, 3)
        normals.scatter_add_(0, indices2, N)
        return normals / _torch.sqrt((normals ** 2).sum(-1, keepdim=True))

    @staticmethod
    def load_obj(path: str, compute_normals = False) -> 'Mesh':
        obj = _vk.load_obj(path)
        vert, indices = _vk.create_mesh(obj, 'po')
        pos = vert[..., 0:3]
        nor = vert[..., 3:6]
        uvs = vert[..., 6:8]
        # pos = obj['buffers']['P']
        # nor = obj['buffers']['N']
        bmin, _ = pos.min(dim=0)
        bmax, _ = pos.max(dim=0)
        pos = (pos - bmin - (bmax - bmin) * 0.5) * 2 / (bmax - bmin).max()
        # indices = obj['buffers']['P_indices']
        normals = nor if nor is not None and not compute_normals else Mesh.compute_normals(pos, indices)
        return Mesh(positions=pos, indices=indices, normals=normals, uvs=uvs)

    @staticmethod
    def box(as_mesh=False):
        return Box(as_mesh)


    def __init__(self, positions: _core.TensorLike | _core.deferred,
                    indices: _core.TensorLike,
                    normals: _core.TensorLike | _core.deferred = None,
                    uvs: _core.TensorLike | _core.deferred = None):
        super().__init__()
        self._positions = _core.ensure_tensor(positions, 2)
        self._indices = indices
        self._normals = _core.ensure_tensor(normals, 2) if normals is not None else None
        self._uvs = _core.ensure_tensor(uvs, 2) if uvs is not None else None

    def patch_info(self, **deferred_parameters) -> dict:
        positions = self._positions if not isinstance(self._positions, _core.deferred) else self._positions.evaluate(**deferred_parameters)
        indices = self._indices
        return dict(
            positions=positions,
            indices=indices,
        )

    def mesh_info(self):
        return dict(
            positions=self._positions,
            indices=self._indices,
            normals=self._normals,
            uvs=self._uvs
        )


class BoxSurfaceMap(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/box_surface.h",
        parameters=dict(
            transform=_torch.Tensor
        )
    )

    def __init__(self, transform: _vk.mat4x3 | _core.TensorLike | _core.deferred, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 6
        if output_dim is None:
            output_dim = 5
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.transform = _core.ensure_tensor(transform, 2)

    def clone(self,
              **kwargs) -> 'BoxSurfaceMap':
        return BoxSurfaceMap(
            self.transform,
            **kwargs)


class SurfacesMap(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/surfaces.h",
        parameters=dict(
            scene_ads=_torch.int64
        )
    )
    def __init__(self,
                 geometries: list[Geometry],
                 transforms: list[_vk.mat4x3 | _core.TensorLike | _core.deferred],
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 6  # ray origin (3) + ray direction (3)
        if output_dim is None:
            # t1 G3 N3 C2 P1
            output_dim = 5
        assert input_dim == 6
        assert output_dim == 5, "SurfacesMap requires output_dim of 5: patch_index, distance, surfel_encode (3)"
        assert len(geometries) == len(transforms), "Number of geometries and transforms must be the same"
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            input_requires_grad=input_requires_grad,
            bw_uses_output=bw_uses_output
        )
        self.geometries = geometries
        self.transforms = transforms
        self.vk_per_geometry_ads_info = [None] * len(self.geometries)
        self.vk_ads_info = None
        self.scene_ads = _vk.wrap_gpu(None)
        self.initialized = False

    def build_geometry_ads(self, geometry: Geometry, vk_ads_info=None, just_update=False, reuse=True, **deferred_parameters: _torch.Tensor) -> dict:
        """
        Builds the acceleration data structures for the geometry.
        Returns the ptr to the built ADS.
        """
        patch_info = geometry.patch_info(**deferred_parameters)
        positions = patch_info['positions']
        indices = patch_info['indices']
        number_of_vertices = positions.shape[0]
        number_of_indices = indices.numel()
        if not reuse or vk_ads_info is None:
            vertex_buffer = _vk.structured_buffer(number_of_vertices, dict(
                position=_vk.vec3
            ), usage=_vk.BufferUsage.RAYTRACING_RESOURCE, memory=_vk.MemoryLocation.GPU)
            index_buffer = _vk.structured_buffer(number_of_indices, element_description=int, usage=_vk.BufferUsage.RAYTRACING_RESOURCE, memory=_vk.MemoryLocation.GPU)
        else:
            assert positions.shape[0] == vk_ads_info['number_of_vertices']
            assert indices.shape[0] == vk_ads_info['number_of_indices']
            vertex_buffer = vk_ads_info['vertex_buffer']
            index_buffer = vk_ads_info['index_buffer']
        vertex_buffer.load(positions)
        index_buffer.load(indices)
        if not reuse or vk_ads_info is None:
            vk_geometry = _vk.triangle_collection()
            vk_geometry.append(vertices=vertex_buffer, indices=index_buffer)
            # Create a bottom ads with the geometry
            vk_geometry_ads = _vk.ads_model(vk_geometry)
            # buffer to scratch ADS builds
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
            vertex_buffer=vertex_buffer,
            index_buffer=index_buffer,
            scratch_buffer=scratch_buffer,
            number_of_vertices=number_of_vertices,
            number_of_indices=number_of_indices,
            vk_geometry_ads_handle = vk_geometry_ads.handle
        )

    def build_ads(self, just_update=False, reuse=True, **deferred_parameters: _torch.Tensor):
        self.initialized = True
        if len(self.geometries) == 0:
            return
        self.vk_per_geometry_ads_info = [
            self.build_geometry_ads(g, info, just_update, reuse, **deferred_parameters) for i, (g, info) in enumerate(zip(self.geometries, self.vk_per_geometry_ads_info))
        ]
        transforms = _torch.zeros(len(self.geometries), 4, 3)
        for i, t in enumerate(self.transforms):
            transforms[i] = t if not isinstance(t, _core.deferred) else t.evaluate(**deferred_parameters)
        if self.vk_ads_info is None or not reuse:
            geometry_ads_ptrs = _torch.zeros(len(self.geometries), 1, dtype=_torch.int64)
            for i, info in enumerate(self.vk_per_geometry_ads_info):
                geometry_ads_ptrs[i] = info['vk_geometry_ads_handle']
            vk_instances = _vk.instance_buffer(len(self.geometries), memory=_vk.MemoryLocation.GPU)
            # Create a top-level ADS with the instances
            vk_scene_ads = _vk.ads_scene(vk_instances)
            # buffer to scratch ADS builds
            scratch_buffer = _vk.scratch_buffer(vk_scene_ads)
        else:
            geometry_ads_ptrs = self.vk_ads_info['geometry_ads_ptrs']
            vk_instances = self.vk_ads_info['vk_instances']
            vk_scene_ads = self.vk_ads_info['vk_scene_ads']
            scratch_buffer = self.vk_ads_info['scratch_buffer']
            # Update or rebuild geometry ADS
            for i, geometry in enumerate(self.geometries):
                geometry.build_ads(just_update=just_update, reuse=True, **deferred_parameters)
        with vk_instances.map('in') as s:
            s.flags = 0
            s.mask8_idx24 = _vk.asint32(0xFF000000)
            # By default, all other values of the instance are filled
            # for instance, transform with identity transform and 0 offset.
            # mask with 255
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
        self.scene_ads = _vk.wrap_gpu(vk_scene_ads.handle)


def compute_surfel_indices(surfels_parameters: str) -> tuple[int, dict[str, int]]:
    """
    Given a string with surfel parameters, returns the total dimension and a dictionary with the indices of each component.
    parameters are treated as a set, order does not matter.
    """
    offset = 0
    indices = dict()
    if 'P' in surfels_parameters:
        indices['POSITION_INDEX'] = offset
        offset += 3
    if 'N' in surfels_parameters:
        indices['SHADING_NORMAL_INDEX'] = offset
        offset += 3
    if 'G' in surfels_parameters:
        indices['GEOMETRY_NORMAL_INDEX'] = offset
        offset += 3
    if 'C' in surfels_parameters:
        indices['UV_INDEX'] = offset
        offset += 2
    if 'T' in surfels_parameters:
        indices['TANGENT_INDEX'] = offset
        offset += 3
    return offset, indices


class SurfelsMap(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/surfels.h",
        parameters=dict(
            patches=['NUMBER_OF_PATCHES', dict(
                __name__='GeometrySurface_PatchInfo',
                positions=_torch.Tensor,
                indices=_torch.Tensor,
                normals=_torch.Tensor,
                uvs=_torch.Tensor,
            )],
            transforms=['NUMBER_OF_PATCHES', _torch.Tensor]
        )
    )

    def __init__(self,
                 geometries: list[Geometry],
                 transforms: list[_vk.mat4x3 | _core.TensorLike | _core.deferred],
                 surfels_parameters: str,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        surfel_dim, surfel_indices = compute_surfel_indices(surfels_parameters)
        if output_dim is None:
            # t1 G3 N3 C2 P1
            output_dim = surfel_dim
        assert output_dim == surfel_dim, f"Output dim {output_dim} does not match the expected size {surfel_dim} from surfels_parameters {surfels_parameters}"
        if input_dim is None:
            input_dim = 5  # patch_index, distance, triangle_index, baricentrics(2)
        assert input_dim == 5
        assert len(geometries) == len(transforms), "Number of geometries and transforms must be the same"
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            input_requires_grad=input_requires_grad,
            bw_uses_output=bw_uses_output,
            NUMBER_OF_PATCHES=max(1, len(geometries)),
            **surfel_indices
        )
        self.geometries = geometries
        for i, t in enumerate(transforms):
            self.transforms[i] = _core.ensure_tensor(t, 2)
        self.surfels_parameters = surfels_parameters
        if (len(self.geometries) == 0):
            # Create a dummy patch to avoid issues with zero patches
            self.patches[0].positions = None
        for i, g in enumerate(self.geometries):
            mi = g.mesh_info()
            pi = self.patches[i]
            pi.positions = mi['positions']
            pi.indices = mi['indices']
            pi.normals = mi['normals']
            pi.uvs = mi['uvs']

    def clone(self,
              **kwargs)->_core.Map:
        return SurfelsMap(
            self.geometries,
            self.transforms,
            self.surfels_parameters,
            **kwargs)


class Lobe:
    """
    Represents atomic BSDF lobe.
    """

    required_surfel_parameters = ""

    def __init__(self, surfel_parameters: str):
        assert (p in self.required_surfel_parameters for p in surfel_parameters)
        self.surfel_parameters = surfel_parameters

    def bsdf(self) -> _core.Map:
        """
        Represents a map that evaluates the cosine weighted BSDF for this lobe.
        win, wout, uv, x -> bsdf value (spectral)
        """
        raise NotImplementedError()

    def bsdf_sampler(self) -> _core.Map:
        """
        Represents a map that samples the cosine weighted BSDF for this lobe.
        win, uv, x -> wout, bsdf value (spectral) / pdf(wout)
        """
        raise NotImplementedError()

    def bsdf_pdf(self) -> _core.Map:
        """
        Represents a map that evaluates the PDF of sampling wout given win for this lobe.
        win, wout, uv, x -> pdf(wout)
        """
        raise NotImplementedError()


class DiffuseLobe(Lobe):

    required_surfel_parameters = "C"  # UV index

    def __init__(self, albedo: _core.MapLike, surfel_parameters: str):
        super().__init__(surfel_parameters)
        self.albedo_map = _core.as_map(albedo)

    class BSDF(_core.Map):
        __extension_info__ = dict(
            path=_core.__INCLUDE_PATH__+"/rendering/bsdf_diffuse.h",
            parameters=dict(
                albedo=_core.Map
            )
        )

        def __init__(self, albedo: _core.MapLike, surfel_parameters: str, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
            surfel_dim, surfel_indices = compute_surfel_indices(surfel_parameters)
            if input_dim is None:
                input_dim = 3 + 3 + surfel_dim # win(3), wout(3), surfel
            assert input_dim == 3 + 3 + surfel_dim
            albedo = _core.as_map(albedo)
            spectral_dim = output_dim if output_dim is not None else albedo.output_dim
            if output_dim is None:
                output_dim = spectral_dim # bsdf value (spectral) * cos_theta(wout)
            super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output, **surfel_indices, SPECTRAL_DIM=spectral_dim)
            self.albedo = albedo.cast(input_dim=2, output_dim=spectral_dim, input_requires_grad=False, bw_uses_output=False)
            self.surfel_parameters = surfel_parameters

        def clone(self,
              **kwargs) -> _core.Map:
            return DiffuseLobe.BSDF(
                self.albedo,
                self.surfel_parameters,
                **kwargs)

    def bsdf(self) -> _core.Map:
        return DiffuseLobe.BSDF(self.albedo_map, self.surfel_parameters)

    class BSDFSampler(_core.Map):
        __extension_info__ = dict(
            path=_core.__INCLUDE_PATH__+"/rendering/bsdf_diffuse_sampler.h",
            parameters=dict(
                albedo=_core.Map
            ),
            stochastic=True
        )

        def __init__(self, albedo: _core.MapLike, surfel_parameters: str, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
            surfel_dim, surfel_indices = compute_surfel_indices(surfel_parameters)
            if input_dim is None:
                input_dim = 3 + surfel_dim # win(3), uv(2), x(3)
            assert input_dim == 3 + surfel_dim
            spectral_dim = output_dim - 4 if output_dim is not None else albedo.output_dim
            if output_dim is None:
                output_dim = 4 + spectral_dim
            super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output, **surfel_indices, SPECTRAL_DIM=spectral_dim)
            self.albedo = albedo.cast(input_dim=2, output_dim=albedo.output_dim, input_requires_grad=False, bw_uses_output=False)
            self.surfel_parameters = surfel_parameters

        def clone(self,
              **kwargs) -> _core.Map:
            return DiffuseLobe.BSDFSampler(
                self.albedo,
                self.surfel_parameters,
                **kwargs)

    def bsdf_sampler(self) -> _core.Map:
        return DiffuseLobe.BSDFSampler(self.albedo_map, self.surfel_parameters)

    class PDF(_core.Map):
        __extension_info__ = dict(
            path=_core.__INCLUDE_PATH__+"/rendering/bsdf_diffuse_pdf.h",
            parameters=dict()
        )

        def __init__(self, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
            if input_dim is None:
                input_dim = 11 # win(3), wout(3), uv(2), x(3)
            if output_dim is None:
                output_dim = 1
            assert input_dim == 11
            assert output_dim == 1
            super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)

        def clone(self,
                  **kwargs)->_core.Map:
            return DiffuseLobe.PDF(
                **kwargs)

    def bsdf_pdf(self) -> _core.Map:
        return DiffuseLobe.PDF()


class FresnelLobe(Lobe):

    required_surfel_parameters = ""

    def __init__(self, refraction_indices: _core.TensorLike, surfel_parameters: str):
        super().__init__(surfel_parameters)
        self.refraction_indices = refraction_indices

    def bsdf(self) -> _core.Map:
        return _core.ZERO

    class BSDFSampler(_core.Map):
        __extension_info__ = dict(
            path=_core.__INCLUDE_PATH__+"/rendering/bsdf_fresnel_sampler.h",
            parameters=dict(
                refraction_indices=_torch.Tensor,
                is_single=int,
            ),
            stochastic=True
        )

        def __init__(self, refraction_indices: _core.TensorLike, surfel_parameters: str, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
            refraction_indices = _core.as_tensor(refraction_indices)
            surfel_dim, surfel_indices = compute_surfel_indices(surfel_parameters)
            if input_dim is None:
                input_dim = 3 + surfel_dim # win(3), uv(2), x(3)
            assert input_dim == 3 + surfel_dim
            spectral_dim = output_dim - 4 if output_dim is not None else None
            if spectral_dim is None:
                if refraction_indices.shape[-1] != 1:
                    spectral_dim = refraction_indices.shape[-1]
            assert spectral_dim is None or refraction_indices.shape[-1] == 1 or refraction_indices.shape[-1] == spectral_dim, "Refraction indices shape does not match spectral dim"
            is_single = 1 if refraction_indices.shape[-1] == 1 else 0
            if output_dim is None and spectral_dim is not None:
                output_dim = 4 + spectral_dim
            super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output, **surfel_indices, SPECTRAL_DIM=spectral_dim)
            self.refraction_indices = refraction_indices
            self.is_single = is_single
            self.surfel_parameters = surfel_parameters

        def clone(self,
              **kwargs) -> _core.Map:
            return FresnelLobe.BSDFSampler(
                self.refraction_indices,
                self.surfel_parameters,
                **kwargs)

    def bsdf_sampler(self) -> _core.Map:
        return FresnelLobe.BSDFSampler(self.refraction_indices, self.surfel_parameters)

    def bsdf_pdf(self) -> _core.Map:
        return _core.ZERO



class MaterialMask:

    required_surfel_parameters = ""

    def mask_to_eval(self, surfel_parameters) -> _core.Map:
        raise NotImplementedError()

    def mask_to_sample(self, surfel_parameters) -> _core.Map:
        raise NotImplementedError()


class Material:
    def flatten_lobes(self, surfel_parameters: str) -> list[list[tuple[list['MaterialMask'], Lobe]]]:
        """
        Returns a list of mixtures
        Each mixture is a list of tuples with weights and a lobe.
        """
        return self.as_layered().flatten_lobes(surfel_parameters)

    def as_layered(self) -> 'LayeredMaterial':
        return LayeredMaterial([self])

    def as_mixture(self) -> 'MixtureMaterial':
        return MixtureMaterial(([], self))

    def required_surfel_parameters(self) -> str:
        raise NotImplementedError()


class UVMaterialMask(MaterialMask):
    required_surfel_parameters = "C"

    def __init__(self, mask: _core.MapLike):
        self.mask = _core.as_map(mask).cast(input_dim=2, output_dim=1)

    def mask_to_eval(self, surfel_parameters) -> _core.Map:
        surfel_dim, surfel_indices = compute_surfel_indices(surfel_parameters)
        return _core.X[6 + surfel_indices['UV_INDEX']:6 + surfel_indices['UV_INDEX'] + 2].then(self.mask)

    def mask_to_sample(self, surfel_parameters) -> _core.Map:
        surfel_dim, surfel_indices = compute_surfel_indices(surfel_parameters)
        return _core.X[3 + surfel_indices['UV_INDEX']:3 + surfel_indices['UV_INDEX'] + 2].then(self.mask)



class MixtureMaterial(Material):
    def __init__(self, *materials: tuple[list[MaterialMask], Material | None]):
        self.basics = []
        for alphas, m in materials:
            alphas = list(filter(None, alphas))
            if isinstance(m, LayeredMaterial):
                assert len(m.layers) == 1, "MixtureMaterial cannot contain LayeredMaterial with more than 1 layer"
                m = m.layers[0]
            if isinstance(m, MixtureMaterial):
                for walphas, wm in m.basics:
                    self.basics.append((alphas + walphas, wm))
            else:
                assert isinstance(m, BasicMaterial), "MixtureMaterial can only contain BasicMaterial or MixtureMaterial"
                self.basics.append((alphas, m))

    def as_mixture(self):
        return self

    def required_surfel_parameters(self) -> str:
        return str(set().union(*(b.required_surfel_parameters+"".join(m.required_surfel_parameters for m in mask_list) for mask_list, b in self.basics)))


class LayeredMaterial(Material):
    def __init__(self, layers: list[Material]):
        self.layers = []
        for l in layers:
            if isinstance(l, LayeredMaterial):
                self.layers.extend(l.layers)
            else:
                self.layers.append(l.as_mixture())

    def as_mixture(self):
        if len(self.layers) == 1:
            return self.layers[0].as_mixture()
        raise NotImplementedError("LayeredMaterial can not be converted to MixtureMaterial if it has more than one layer.")

    def as_layered(self):
        return self

    def flatten_lobes(self, surfel_parameters)  -> list[list[tuple[list[MaterialMask], Lobe]]]:
        return [[(a, b.get_lobe(surfel_parameters)) for a, b in m.basics] for m in self.layers]

    def required_surfel_parameters(self) -> str:
        return str(set().union(*(m.required_surfel_parameters() for m in self.layers)))


class BasicMaterial(Material):
    def get_lobe(self, surfel_parameters: str):
        raise NotImplementedError()


class FresnelMaterial(BasicMaterial):
    def __init__(self, refraction_indices: _core.TensorLike = 1.5):
        self.refraction_indices = refraction_indices

    def get_lobe(self, surfel_parameters: str):
        return FresnelLobe(refraction_indices=self.refraction_indices, surfel_parameters=surfel_parameters)

    def required_surfel_parameters(self) -> str:
        return FresnelLobe.required_surfel_parameters


class DiffuseMaterial(BasicMaterial):
    def __init__(self, albedo: _core.MapLike):
        self.albedo = albedo

    def get_lobe(self, surfel_parameters: str):
        return DiffuseLobe(albedo=self.albedo, surfel_parameters=surfel_parameters)

    def required_surfel_parameters(self) -> str:
        return DiffuseLobe.required_surfel_parameters


def create_mixture_sampler(
        map_ids: list[int],
        mask_id: int,
        max_repeats: int,
        max_masks: int
    ) -> type:
    """
    Given a list of map IDs and a mask ID, creates a MixtureSampler class that combines maps of such types.
    The probability of evaluating a map is the product of its masks normalized.
    If mask_id is 0, no masks are used and all weights are considered equal.
    max_repeats: maximum number of times a map type can be repeated in the mixture.
    max_masks: maximum number of masks per map.
    """
    NUMBER_OF_MAPS = len(map_ids)  # different map types
    MAX_NUMBER_OF_MASKS = max_masks
    MAX_REPEATS = max_repeats
    parameters = dict(
        maps=['NUMBER_OF_MAPS', ['MAX_REPEATS', _torch.int64]], # treated as pointers to maps
        masks=['NUMBER_OF_MAPS', ['MAX_REPEATS', ['MAX_NUMBER_OF_MASKS', _torch.int64]]], # treated as pointers to maps
        num_repeats=['NUMBER_OF_MAPS', int],
        num_masks=['NUMBER_OF_MAPS', ['MAX_REPEATS', int]],
    )
    mixture_code = """
FORWARD {
    float total_weight = 0.0;
    """
    # compute total weight among all material types
    for k in range(NUMBER_OF_MAPS):
        mixture_code += f"""
    #ifdef MASK_TYPE
    for (int i=0; i < parameters.num_repeats[{k}]; i++)
    {{
        float current_weight = 1.0;
        for (int j=0; j < parameters.num_masks[{k}][i]; j++) {{
            float mask_value[1];
            forward(CAST(MASK_TYPE, (parameters.masks[{k}][i][j])), _input, mask_value);
            current_weight *= mask_value[0];
        }}
        total_weight += current_weight;
    }}
    #else
    total_weight += float(parameters.num_repeats[{k}]);
    #endif
    """
    # select one material based on weights and evaluate its transport
    mixture_code += """
    float sel = random() * total_weight;
    """
    for k in range(NUMBER_OF_MAPS):
        mixture_code += f"""
    for (int i=0; i < parameters.num_repeats[{k}]; i++)
    {{
        float current_weight = 1.0;
        #ifdef MASK_TYPE
        for (int j=0; j < parameters.num_masks[{k}][i]; j++) {{
            float mask_value[1];
            forward(CAST(MASK_TYPE, parameters.masks[{k}][i][j]), _input, mask_value);
            current_weight *= mask_value[0];
        }}
        #endif
        if (sel < current_weight) {{
            forward(CAST(MAP_TYPE_{k}, parameters.maps[{k}][i]), _input, _output);
            return;
        }}
        sel -= current_weight;
    }}
    """
    mixture_code += """
}
    """

    class MixtureSampler(_core.Map):
        __extension_info__ = dict(
            code=mixture_code,
            parameters=parameters,
            stochastic=True
        )
        def __init__(self, maps: list[_core.Map], masks: list[list[_core.Map]], input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
            assert len(maps) == len(masks), "Number of maps and masks lists must be the same"
            assert all(not m.is_generic for m in maps), "All maps must be concrete. Use cast() to convert generic maps to concrete ones."
            if input_dim is None:
                input_dim = maps[0].input_dim
            assert all(m.input_dim == input_dim for m in maps), "All maps must have the same input_dim"
            if output_dim is None:
                output_dim = maps[0].output_dim
            assert all(m.output_dim == output_dim for m in maps), "All maps must have the same output_dim"
            maps = [m.cast(input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output) for m in maps]
            masks = [[msk.cast(input_dim=input_dim, output_dim=1, input_requires_grad=False, bw_uses_output=False) for msk in mask_list] for mask_list in masks]
            self.original_maps = maps
            self.original_masks = masks
            partitioned_maps = [[] for _ in range(NUMBER_OF_MAPS)]
            partitioned_masks = [[] for _ in range(NUMBER_OF_MAPS)]
            types = {}
            for m, masks in zip(maps, masks):
                compute_id = m.rdv_kernel_id  # granted due to concrete check above
                index = map_ids.index(compute_id)
                partitioned_maps[index].append(m)
                partitioned_masks[index].append(masks)
            if mask_id != 0:
                types = {'MASK_TYPE': _core._DispatcherEngine.__KERNELS__[mask_id].codename}
            for k in range(NUMBER_OF_MAPS):
                ki = _core._DispatcherEngine.__KERNELS__[map_ids[k]]
                types[f'MAP_TYPE_{k}'] = ki.codename
            super().__init__(
                input_dim=input_dim,
                output_dim=output_dim,
                input_requires_grad=input_requires_grad,
                bw_uses_output=bw_uses_output,
                NUMBER_OF_MAPS=NUMBER_OF_MAPS,
                MAX_REPEATS=MAX_REPEATS,
                MAX_NUMBER_OF_MASKS=max(1, MAX_NUMBER_OF_MASKS),
                **types)
            for k, (maps, masks) in enumerate(zip(partitioned_maps, partitioned_masks)):
                self.num_repeats[k] = len(maps)
                self.maps[k][0] = _vk.wrap_gpu(None)  # in case of zero maps
                self.masks[k][0][0] = _vk.wrap_gpu(None)  # in case of zero masks
                for i, (map, mask_list) in enumerate(zip(maps, masks)):
                    self.maps[k][i] = map
                    self.num_masks[k][i] = len(mask_list)
                    self.masks[k][i][0] = _vk.wrap_gpu(None)  # in case of zero masks
                    for j, mask in enumerate(mask_list):
                        self.masks[k][i][j] = mask

        def clone(self,
                  **kwargs) -> _core.Map:
            return MixtureSampler(
                self.original_lobes,
                self.original_masks,
                **kwargs)

        def extended_dependencies(self):
            return map_ids + ([mask_id] if mask_id != 0 else [])

    return MixtureSampler


def create_mixture(
        map_ids: list[int],
        mask_id: int,
        max_repeats: int,
        max_masks: int
    ) -> type:
    """
    Given a list of map IDs and a mask ID, creates a Mixture class that combines maps of such types.
    The final eval of the mixture is the sum of evaluating all maps multiplied to the product of their masks normalized.
    If mask_id is 0, no masks are used and all weights are considered equal.
    max_repeats: maximum number of times a map type can be repeated in the mixture.
    max_masks: maximum number of masks per map.
    """
    NUMBER_OF_MAPS = len(map_ids)  # different map types
    MAX_NUMBER_OF_MASKS = max_masks
    MAX_REPEATS = max_repeats
    parameters = dict(
        maps=['NUMBER_OF_MAPS', ['MAX_REPEATS', _torch.int64]], # treated as pointers to maps
        masks=['NUMBER_OF_MAPS', ['MAX_REPEATS', ['MAX_NUMBER_OF_MASKS', _torch.int64]]], # treated as pointers to maps
        num_repeats=['NUMBER_OF_MAPS', int],
        num_masks=['NUMBER_OF_MAPS', ['MAX_REPEATS', int]],
    )
    mixture_code = """
FORWARD {
    float total_weight = 0.0;
    for (int i=0; i<OUTPUT_DIM; i++) {
        _output[i] = 0.0; // initialize output
    }
    float _temp_output[OUTPUT_DIM]; 
    """
    for k in range(NUMBER_OF_MAPS):
        mixture_code += f"""
    for (int i=0; i < parameters.num_repeats[{k}]; i++)
    {{
        float current_weight = 1.0;
        #ifdef MASK_TYPE
        for (int j=0; j < parameters.num_masks[{k}][i]; j++) {{
            float mask_value[1];
            forward(CAST(MASK_TYPE, parameters.masks[{k}][i][j]), _input, mask_value);
            current_weight *= mask_value[0];
        }}
        #endif
        forward(CAST(MAP_TYPE_{k}, parameters.maps[{k}][i]), _input, _temp_output);
        for (int j=0; j<OUTPUT_DIM; j++) {{
            _output[j] += current_weight * _temp_output[j];
        }}
        total_weight += current_weight;
    }}
    """
    mixture_code += """
    // normalize output
    if (total_weight > 0.0) 
        for (int i=0; i<OUTPUT_DIM; i++) 
            _output[i] /= total_weight;
}
    """

    class Mixture(_core.Map):
        __extension_info__ = dict(
            code=mixture_code,
            parameters=parameters,
        )
        def __init__(self, maps: list[_core.Map], masks: list[list[_core.Map]], input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
            assert len(maps) == len(masks), "Number of maps and masks lists must be the same"
            assert all(not m.is_generic for m in maps), "All maps must be concrete. Use cast() to convert generic maps to concrete ones."
            if input_dim is None:
                input_dim = maps[0].input_dim
            assert all(m.input_dim == input_dim for m in maps), "All maps must have the same input_dim"
            if output_dim is None:
                output_dim = maps[0].output_dim
            assert all(m.output_dim == output_dim for m in maps), "All maps must have the same output_dim"
            maps = [m.cast(input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output) for m in maps]
            masks = [[msk.cast(input_dim=input_dim, output_dim=1, input_requires_grad=False, bw_uses_output=False) for msk in mask_list] for mask_list in masks]
            self.original_maps = maps
            self.original_masks = masks
            partitioned_maps = [[] for _ in range(NUMBER_OF_MAPS)]
            partitioned_masks = [[] for _ in range(NUMBER_OF_MAPS)]
            types = {}
            for m, masks in zip(maps, masks):
                compute_id = m.rdv_kernel_id  # granted due to concrete check above
                index = map_ids.index(compute_id)
                partitioned_maps[index].append(m)
                partitioned_masks[index].append(masks)
            if mask_id != 0:
                types = {'MASK_TYPE': _core._DispatcherEngine.__KERNELS__[mask_id].codename}
            for k in range(NUMBER_OF_MAPS):
                ki = _core._DispatcherEngine.__KERNELS__[map_ids[k]]
                types[f'MAP_TYPE_{k}'] = ki.codename
            super().__init__(
                input_dim=input_dim,
                output_dim=output_dim,
                input_requires_grad=input_requires_grad,
                bw_uses_output=bw_uses_output,
                NUMBER_OF_MAPS=NUMBER_OF_MAPS,
                MAX_REPEATS=MAX_REPEATS,
                MAX_NUMBER_OF_MASKS=max(1, MAX_NUMBER_OF_MASKS),
                **types)
            for k, (maps, masks) in enumerate(zip(partitioned_maps, partitioned_masks)):
                self.num_lobes[k] = len(maps)
                self.lobes[k][0] = _vk.wrap_gpu(None)  # in case of zero maps
                self.masks[k][0][0] = _vk.wrap_gpu(None)  # in case of zero masks
                for i, (map, mask_list) in enumerate(zip(maps, masks)):
                    self.lobes[k][i] = map
                    self.num_masks[k][i] = len(mask_list)
                    self.masks[k][i][0] = _vk.wrap_gpu(None)  # in case of zero masks
                    for j, mask in enumerate(mask_list):
                        self.masks[k][i][j] = mask

        def clone(self,
                  **kwargs) -> _core.Map:
            return Mixture(
                self.original_lobes,
                self.original_masks,
                **kwargs)

        def extended_dependencies(self):
            return map_ids + ([mask_id] if mask_id != 0 else [])

    return Mixture


def create_medium_composition_integrator(media_count: int, map_ids: list[int], spectral_dim: int):
    SPECTRAL_DIM = spectral_dim
    parameters = dict(
        **{f'event_sampler_{k}': _core.Map for k in range(media_count)},
        **{f'scattering_sampler_{k}': _core.Map for k in range(media_count)},
    )
    composition_code = """
FORWARD {
    int medium_index = -1;
    float closest_event[1 + MAX_EVENT_INFO_DIM];
    """
    for k in range(media_count):
        composition_code += f"""
    float event_sampler_out[1 + EVENT_INFO_DIM_{k}];
    forward(parameters.event_sampler_{k}, _input, event_sampler_out);
    if (event_sampler_out[0] < _input[6] && closest_event[0] > event_sampler_out[0]) {{
        for (int i=0; i<1 + EVENT_INFO_DIM_{k}; i++) closest_event[i] = event_sampler_out[i];
        medium_index = {k};
    }}
        """
    composition_code += """
    if (medium_index == -1) // no medium event, return transmittance to t_max
    {
        // t_hit = t_max
        // wout = win (no scattering)
        // weight = 1.0 (for all spectral components)
        _output[0] = _input[6]; // t_max
        _output[1] = _input[3]; // win
        _output[2] = _input[4]; // win
        _output[3] = _input[5]; // win
        for (int i=0; i<SPECTRAL_DIM; i++) _output[4 + i] = 1.0; // weight
        return;
    }
    """
    for k in range(media_count):
        composition_code += f"""
    if ({k} == medium_index) {{
        float scattering_sampler_output[SPECTRAL_DIM + 3 + 1];
        float scattering_sampler_input[6 + EVENT_INFO_DIM_{k}];
        scattering_sampler_input[0] = _input[0] + closest_event[0] * _input[3]; // x
        scattering_sampler_input[1] = _input[1] + closest_event[0] * _input[4]; // x
        scattering_sampler_input[2] = _input[2] + closest_event[0] * _input[5]; // x
        scattering_sampler_input[3] = _input[3]; // win
        scattering_sampler_input[4] = _input[4]; // win
        scattering_sampler_input[5] = _input[5]; // win
        for (int i=0; i<EVENT_INFO_DIM_{k}; i++) scattering_sampler_input[6 + i] = closest_event[1 + i];
        forward(parameters.scattering_sampler_{k}, scattering_sampler_input, scattering_sampler_output);
        _output[0] = closest_event[0]; // hit_t
        _output[1] = scattering_sampler_output[0]; // win
        _output[2] = scattering_sampler_output[1]; // win
        _output[3] = scattering_sampler_output[2]; // win
        for (int i=0; i<SPECTRAL_DIM; i++) _output[4 + i] = scattering_sampler_output[i+3]; // weight
    }}
        """
    composition_code += """
}
    """

    class CompositionIntegrator(_core.Map):
        __extension_info__ = dict(
            code=composition_code,
            parameters=parameters,
            stochastic=True
        )
        def __init__(self, event_samplers: list[_core.Map], scattering_samplers: list[_core.Map], input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
            assert len(event_samplers) == media_count
            assert len(scattering_samplers) == media_count
            if input_dim is None:
                input_dim = 7
            if output_dim is None:
                output_dim = 4 + SPECTRAL_DIM
            super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output, SPECTRAL_DIM=SPECTRAL_DIM, **{f'EVENT_INFO_DIM_{k}': event_samplers[k].output_dim - 1 for k in range(media_count)})
            for k in range(media_count):
                setattr(self, f'event_sampler_{k}', event_samplers[k])
                setattr(self, f'scattering_sampler_{k}', scattering_samplers[k])

        def clone(self,
                    **kwargs) -> _core.Map:
                return CompositionIntegrator(
                    [getattr(self, f'event_sampler_{k}') for k in range(media_count)],
                    [getattr(self, f'scattering_sampler_{k}') for k in range(media_count)],
                    **kwargs)

        def extended_dependencies(self):
            return map_ids

    return CompositionIntegrator


class Density:
    def get_exitinction_map(self) -> _core.Map:
        raise NotImplementedError()

    def get_majorant_map(self) -> _core.Map:
        raise NotImplementedError()


class ScatteringKernel:
    def get_scattering_albedo_map(self) -> _core.Map:
        raise NotImplementedError()

    def get_phase_function_map(self) -> _core.Map:
        raise NotImplementedError()

    def get_phase_function_sampler_map(self)->_core.Map:
        raise NotImplementedError()


class MediumScatterer:
    def event_sampler(self) -> _core.Map:
        """
        Given x, w, t_max -> t, event_info (7 -> 1 + EVENT_INFO_DIM)
        """
        raise NotImplementedError()

    def scattering_sampler(self) -> _core.Map:
        """
        Given x_t, w, event_info -> radiance scattered (SPECTRAL_DIM), wout, pdf(wout)
        """
        raise NotImplementedError()

    def scattering_kernel(self) -> _core.Map:
        """
        Given x_t, win, wout, event_info -> radiance scattered (SPECTRAL_DIM)
        """
        raise NotImplementedError()

    def scattering_pdf(self) -> _core.Map:
        """
        Given x_t, win, wout, event_info -> pdf(wout)
        """
        raise NotImplementedError()

    def transmittance(self):
        """
        Given x, w, t_max -> transmittance to t_max
        """
        raise NotImplementedError()


class MediumEmitter:
    def emission_integral(self) -> _core.Map:
        """
        Given x, w, t_max -> emitted radiance (SPECTRAL_DIM), transmittance to t_max
        """
        raise NotImplementedError()

    def transmittance(self):
        """
        Given x, w, t_max -> transmittance to t_max
        """
        raise NotImplementedError()


class Medium:
    def scatterers(self) -> list[MediumScatterer]:
        pass

    def emitters(self) -> list[MediumEmitter]:
        pass


class ScatteringMedium(Medium):
    @cached_property
    def scatterer(self) -> MediumScatterer:
        raise NotImplementedError()

    def scatterers(self) -> list[MediumScatterer]:
        return [self.scatterer]

    def emitters(self) -> list[MediumEmitter]:
        return []


class EmissionMedium(Medium):
    @cached_property
    def emitter(self) -> MediumEmitter:
        raise NotImplementedError()

    def scatterers(self) -> list[MediumScatterer]:
        return []

    def emitters(self) -> list[MediumEmitter]:
        return [self.emitter]


class CompositeMedium(Medium):
    def __init__(self, media: list[Medium]):
        self._scatterers = []
        self._emitters = []
        for m in media:
            self._scatterers.extend(m.scatterers())
            self._emitters.extend(m.emitters())

    def scatterers(self) -> list[MediumScatterer]:
        return self._scatterers

    def emitters(self) -> list[MediumEmitter]:
        return self._emitters


class RatioTrackingMediumTransmittance(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + "/rendering/medium_transmittance_rt.h",
        parameters = dict(
            extinction = _core.Map,
            majorant = _core.Map,
            transform = _torch.Tensor
        ),
    )

    def __init__(self, extinction: _core.MapLike, majorant: _core.MapLike, transform: _core.TensorLike | _core.deferred, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 7
        if output_dim is None:
            output_dim = 1
        assert input_dim == 7
        assert output_dim == 1
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.extinction = extinction
        self.majorant = majorant
        self.transform = _core.ensure_tensor(transform, 2)

    def clone(self,
              **kwargs) -> _core.Map:
        return RatioTrackingMediumTransmittance(
            self.extinction,
            self.majorant,
            self.transform,
            **kwargs)


class HGMediumScatterer(MediumScatterer):
    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 transform: _core.TensorLike | _core.deferred):
        self.extinction = _core.as_map(extinction).cast(input_dim=3, output_dim=1)
        self.majorant = _core.as_map(majorant).cast(input_dim=6, output_dim=2)
        self.scattering_albedo = _core.as_map(scattering_albedo)
        self.anisotropy = _core.as_map(anisotropy).cast(input_dim=3, output_dim=1)
        self.transform = transform

    class DeltatrackingEventSampler(_core.Map):
        __extension_info__ = dict(
            path=_core.__INCLUDE_PATH__ + "/rendering/medium_event_sampler_dt.h",
            parameters = dict(
                extinction=_core.Map,
                majorant=_core.Map,
                transform=_torch.Tensor
            ),
            stochastic=True
        )
        def __init__(self, extinction, majorant, transform: _core.deferred | _core.TensorLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
            if input_dim is None:
                input_dim = 7
            if output_dim is None:
                output_dim = 1
            assert input_dim == 7
            assert output_dim == 1
            super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
            self.extinction = extinction
            self.majorant = majorant
            self.transform = _core.ensure_tensor(transform, 2)

    def event_sampler(self) -> _core.Map:
        return HGMediumScatterer.DeltatrackingEventSampler(
            self.extinction,
            self.majorant,
            self.transform)

    class HGScatteringSampler(_core.Map):
        __extension_info__ = dict(
            path=_core.__INCLUDE_PATH__ + "/rendering/medium_scattering_sampler_hg.h",
            parameters = dict(
                scattering_albedo=_core.Map,
                anisotropy=_core.Map
            ),
            stochastic=True
        )

        def __init__(self, scattering_albedo, anisotropy, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
            spectral_dim = output_dim - 4 if output_dim is not None else None
            spectral_dim = scattering_albedo.output_dim if spectral_dim is None else spectral_dim
            if input_dim is None:
                input_dim = 6  # x_t(3), win(3)
            assert input_dim == 6
            if output_dim is None and spectral_dim is not None:
                output_dim = 3 + spectral_dim + 1 # wout(3), spectral(SPECTRAL_DIM), pdf(1)
            if spectral_dim is not None:
                assert output_dim == 3 + spectral_dim + 1
            super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output, SPECTRAL_DIM=spectral_dim)
            self.scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
            self.anisotropy = anisotropy.cast(input_dim=3, output_dim=1, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)

    def scattering_sampler(self) -> _core.Map:
        return HGMediumScatterer.HGScatteringSampler(self.scattering_albedo, self.anisotropy)

    class HGScatteringPDF(_core.Map):
        __extension_info__ = dict(
            path=_core.__INCLUDE_PATH__ + "/rendering/medium_scattering_pdf_hg.h",
            parameters = dict(
                anisotropy=_core.Map
            ),
        )

        def __init__(self, anisotropy, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
            if input_dim is None:
                input_dim = 9  # x_t(3), win(3), wout(3)
            if output_dim is None:
                output_dim = 1
            super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
            self.anistropy = anisotropy

    def scattering_pdf(self) -> _core.Map:
        return HGMediumScatterer.HGScatteringPDF(self.anisotropy)

    class HGScattering(_core.Map):
        __extension_info__ = dict(
            path=_core.__INCLUDE_PATH__ + "/rendering/medium_scattering_hg.h",
            parameters = dict(
                scattering_albedo=_core.Map,
                anisotropy=_core.Map
            )
        )
        def __init__(self, scattering_albedo, anisotropy, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
            if input_dim is None:
                input_dim = 9
            assert input_dim == 9
            if output_dim is None:
                output_dim = scattering_albedo.output_dim
            assert output_dim == scattering_albedo.output_dim
            super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
            self.scattering_albedo = scattering_albedo
            self.anisotropy = anisotropy

    def scattering_kernel(self) -> _core.Map:
        return HGMediumScatterer.HGScattering(self.scattering_albedo, self.anisotropy)

    def transmittance(self) -> _core.Map:
        return RatioTrackingMediumTransmittance(self.extinction, self.majorant, self.transform)


class HGMedium(ScatteringMedium):

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike = None,
                 anisotropy: _core.MapLike = None,
                 transform: _core.TensorLike | _core.deferred | None = None,
                 ):
        self.extinction = extinction
        self.majorant = majorant
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy
        self.transform = transform if transform is not None else _vk.mat4x3.trs()

    @cached_property
    def scatterer(self):
        return HGMediumScatterer(
            self.extinction,
            self.majorant,
            self.scattering_albedo,
            self.anisotropy,
            self.transform,
        )




# class Medium:
#     def __init__(self,
#                  density: Density,
#                  scattering_kernel: ScatteringKernel = None,
#                  emission: VolumeEmission = None):
#         self.density = density
#         self.scattering_kernel = scattering_kernel
#         self.emission = emission
#
#
# class GridDensity(Density):
#     def __init__(self,
#                  density: _core.TensorLike | _core.deferred
#         ):
#         self.density = _core.ensure_tensor(density, 4)
#         self.majorant_tensor = _vk.tensor_from([1.0 if isinstance(self.density, _core.deferred) else self.density.max().item(), 1000000.0])
#
#     def update(self, **deferred_parameters):
#         density = self.density if not isinstance(self.density, _core.deferred) else self.density.evaluate(**deferred_parameters)
#         self.majorant_tensor[0] = density.max().item()
#
#     def get_exitinction_map(self) -> _core.Map:
#         return _core.Sample3DMap(self.density)
#
#     def get_majorant_map(self) -> _core.Map:
#         return _core.ConstantMap(self.majorant_tensor)
#
#
# class GridScatteringKernel(ScatteringKernel):
#     def __init__(self,
#                  g_factor: _core.TensorLike | _core.deferred,
#                  scattering_albedo: _core.TensorLike | _core.deferred):
#         self.g_factor = _core.ensure_tensor(g_factor, 4)
#         self.scattering_albedo = _core.ensure_tensor(scattering_albedo, 4)
#
#     def get_scattering_albedo_map(self) -> _core.Map:
#         return _core.Sample3DMap(self.scattering_albedo)
#
#     def get_phase_function_map(self) -> _core.Map:
#         return HGPhaseFunction( _core.Sample3DMap(self.g_factor))
#
#     def get_phase_function_sampler_map(self) -> _core.Map:
#         return HGPhaseFunctionSampler( _core.Sample3DMap(self.g_factor))
#
#
# class ConstantScatteringKernel(ScatteringKernel):
#     def __init__(self, g_factor: _core.TensorLike = None, scattering_albedo: _core.TensorLike = 0.0):
#         g_factor = _core.as_tensor(g_factor)
#         self.g_factor = _core.ensure_tensor(g_factor, map_dim=1)
#         scattering_albedo = _core.as_tensor(scattering_albedo)
#         self.scattering_albedo = _core.ensure_tensor(scattering_albedo, map_dim=1)
#
#     def get_scattering_albedo_map(self) -> _core.Map:
#         return _core.ConstantMap(self.scattering_albedo)
#
#     def get_phase_function_map(self) -> _core.Map:
#         return HGPhaseFunction(self.g_factor)
#
#     def get_phase_function_sampler_map(self) -> _core.Map:
#         return HGPhaseFunctionSampler(self.g_factor)
#
#
# class GridVolumeEmission(VolumeEmission):
#     def __init__(self,
#                  emission: _core.TensorLike | _core.deferred):
#         self.emission = _core.ensure_tensor(emission, 4)


class Visual:
    def __init__(self,
                 geometry: Geometry,
                 material: Material | None = None,
                 transform: _vk.mat4x3 | _core.TensorLike | _core.deferred = _vk.mat4x3.trs(),
                 medium: Medium = None,
                 parent: 'Visual' = None  # used to identify outside medium == parent.medium
                 ):
        self.material = material
        self.geometry = geometry
        self.transform = transform
        self.medium = medium
        self.parent = parent



class SceneTransmittance(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + "/rendering/scene_transmittance.h",
        parameters=dict(
            surfaces=_core.Map,
            has_surface=['NUMBER_OF_PATCHES', int],
            outside_medium=['NUMBER_OF_PATCHES', int],  # < NUMBER_OF_MEDIA
            inside_medium=['NUMBER_OF_PATCHES', int],  # < NUMBER_OF_MEDIA
            medium_transmittance=['NUMBER_OF_MEDIA', _core.Map],
        )
    )

    def __init__(self,
                 number_of_patches: int | None = None,
                 surfaces: _core.MapLike = None,
                 has_surface: list[int] = None,
                 outside_medium: list[int] = None,
                 inside_medium: list[int] = None,
                 medium_transmittance: list[_core.MapLike] = None,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 6  # ray origin (3) + ray direction (3)
        assert input_dim == 6
        if medium_transmittance is not None:
            medium_transmittance = [_core.as_map(mt) for mt in medium_transmittance]
        surfaces = _core.as_map(surfaces, default=_core.ConstantMap([
            _np.float32(_np.inf),
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0,
            _np.frombuffer(_np.int32(-1).tobytes(), dtype=_np.float32)[0]
        ]))
        if output_dim is None:
            output_dim = 1
        medium_transmittance = [m.cast(input_dim=7, output_dim=1) for m in
                            medium_transmittance] if medium_transmittance is not None else None
        assert has_surface is not None or number_of_patches is not None or inside_medium is not None, "Either has_surface or number_of_patches must be provided"
        if number_of_patches is None:
            if has_surface is not None:
                number_of_patches = len(has_surface)
            else:
                number_of_patches = len(inside_medium)
        if has_surface is None:
            has_surface = [0] * number_of_patches
        outside_medium = outside_medium if outside_medium is not None else [-1] * number_of_patches
        inside_medium = inside_medium if inside_medium is not None else [-1] * number_of_patches
        assert number_of_patches == len(outside_medium) == len(inside_medium) == len(has_surface), "Length of outside_medium, inside_medium, has_surface must be equal to number_of_patches"
        # avoid empty array with dummy maps
        if medium_transmittance is None or len(medium_transmittance) == 0:
            medium_transmittance = [_core.ONE.cast(input_dim=7, output_dim=1)]
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad,
            bw_uses_output=bw_uses_output,
            NUMBER_OF_PATCHES=number_of_patches,
            NUMBER_OF_MEDIA=len(medium_transmittance),
        )
        self.surfaces = surfaces.cast(6, 5, input_requires_grad, bw_uses_output)
        for i, value in enumerate(has_surface):
            self.has_surface[i] = value
        for i, map in enumerate(medium_transmittance):
            self.medium_transmittance[i] = map
        for i, idx in enumerate(outside_medium):
            self.outside_medium[i] = idx
        for i, idx in enumerate(inside_medium):
            self.inside_medium[i] = idx

    def clone(self,
              **kwargs) -> _core.Map:
        return SceneTransmittance(
            number_of_patches=len(self.surface_integrator_indices),
            surfaces=self.surfaces,
            has_surface=self.has_surface,
            outside_medium=self.outside_medium,
            inside_medium=self.inside_medium,
            medium_transmittance=self.medium_transmittance,
            **kwargs
        )


class ScenePathIntegrator(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/scene_path_integrator.h",
        parameters=dict(
            surfaces=_core.Map,
            surfels=_core.Map,
            environment=_core.Map,
            surface_integrators=['NUMBER_OF_MATERIALS', _core.Map],
            medium_integrators=['NUMBER_OF_MEDIA', _core.Map],
            surface_integrator_indices=['NUMBER_OF_PATCHES', int], # < NUMBER_OF_SURFACE_CONTRIBUTION
            outside_medium=['NUMBER_OF_PATCHES', int], # < NUMBER_OF_MEDIA
            inside_medium=['NUMBER_OF_PATCHES', int], # < NUMBER_OF_MEDIA
        )
    )

    def __init__(self,
                    number_of_patches: int,
                    surfel_dim: int,
                    spectral_dim: int,
                    path_state_dim: int,
                    surfaces: _core.MapLike = None,
                    surfels: _core.MapLike = None,
                    environment: _core.MapLike = None,
                    surface_integrators: list[_core.MapLike] = None,
                    medium_integrators: list[_core.MapLike] = None,
                    surface_integrator_indices: list[int] = None,
                    outside_medium: list[int] = None,
                    inside_medium: list[int] = None,
                    input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 6  # ray origin (3) + ray direction (3)
        assert input_dim == 6
        if medium_integrators is None:
            medium_integrators = []
        medium_integrators = [_core.as_map(mi) for mi in medium_integrators]
        if surface_integrators is None:
            surface_integrators = []
        surface_integrators = [_core.as_map(si) for si in surface_integrators]
        has_environment = environment is not None and environment != _core.ZERO
        environment = _core.as_map(environment, default=_core.ZERO)
        surfaces = _core.as_map(surfaces, default=_core.ConstantMap([
            _np.frombuffer(_np.int32(-1).tobytes(), dtype=_np.float32)[0],
            _np.float32(_np.inf),
            _np.frombuffer(_np.int32(-1).tobytes(), dtype=_np.float32)[0],
            0.0, 0.0
        ]))
        surfels = _core.as_map(surfels)
        if output_dim is None:
            output_dim = spectral_dim + path_state_dim
        surface_integrators = [s.cast(input_dim=6 + surfel_dim + path_state_dim, output_dim=spectral_dim * 2 + 6 + path_state_dim) for s in surface_integrators] if surface_integrators is not None else None
        medium_integrators = [m.cast(input_dim=7 + path_state_dim, output_dim=spectral_dim * 2 + 6 + path_state_dim) for m in medium_integrators] if medium_integrators is not None else None
        surface_integrator_indices = surface_integrator_indices if surface_integrator_indices is not None else [-1] * number_of_patches
        outside_medium = outside_medium if outside_medium is not None else [-1] * number_of_patches
        inside_medium = inside_medium if inside_medium is not None else [-1] * number_of_patches
        assert number_of_patches == len(outside_medium) == len(inside_medium) == len(surface_integrator_indices), "Length of surface_integrator_indices, outside_medium, inside_medium must be equal to number_of_patches"
        assert all(i < len(surface_integrators) for i in surface_integrator_indices), "surface_contribution_indices contains invalid indices"
        assert all(i < len(medium_integrators) for i in inside_medium), "inside_medium contains invalid indices"
        assert all(i < len(medium_integrators) for i in outside_medium), "outside_medium contains invalid indices"
        # avoid empty array with dummy maps
        if len(medium_integrators) == 0:
            medium_integrators = [_core.ZERO.cast(input_dim=7 + path_state_dim, output_dim=spectral_dim * 2 + 6 + path_state_dim)]
        if len(surface_integrators) == 0:
            surface_integrators = [_core.ZERO.cast(input_dim=6 + surfel_dim + path_state_dim, output_dim=spectral_dim * 2 + 6 + path_state_dim)]
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            NUMBER_OF_PATCHES=number_of_patches,
            NUMBER_OF_MATERIALS=len(surface_integrators),
            NUMBER_OF_MEDIA=len(medium_integrators),
            **({ 'HAS_ENVIRONMENT': 1} if has_environment else {}),
            SPECTRAL_DIM=spectral_dim,
            SURFEL_DIM=surfel_dim,
            PATH_STATE_DIM=path_state_dim,
        )
        self.number_of_patches = number_of_patches
        self.surfel_dim = surfel_dim
        self.spectral_dim = spectral_dim
        self.path_state_dim = path_state_dim
        self.surfaces = surfaces.cast(6, 5, False, False)
        self.surfels = surfels.cast(5, surfel_dim, False, False)
        self.environment = environment.cast(3, spectral_dim, input_requires_grad, bw_uses_output)
        for i, idx in enumerate(surface_integrator_indices):
            self.surface_integrator_indices[i] = idx
        for i, idx in enumerate(outside_medium):
            self.outside_medium[i] = idx
        for i, idx in enumerate(inside_medium):
            self.inside_medium[i] = idx
        for i, I in enumerate(surface_integrators):
            self.surface_integrators[i] = I
        for i, I in enumerate(medium_integrators):
            self.medium_integrators[i] = I

    def clone(self,
              **kwargs) -> _core.Map:
        return  ScenePathIntegrator(
            number_of_patches=self.number_of_patches,
            surfel_dim=self.surfel_dim,
            spectral_dim=self.spectral_dim,
            path_state_dim=self.path_state_dim,
            surfaces=self.surfaces,
            environment=self.environment,
            outside_medium=self.outside_medium,
            inside_medium=self.inside_medium,
            surface_integrators=self.surface_integrators,
            medium_integrators=self.medium_integrators,
            **kwargs
        )


class ScenePathIntegratorObjects:
    medium_integrators = None
    surface_integrators = None
    path_state_dim = 0
    surfel_parameters = ""
    surfel_dim = 0

    def __init__(self,
                 surface_integrators=None,
                 medium_integrators=None,
                 path_state_dim=0,
                 surfel_parameters="",
                 surfel_dim=0):
        self.surface_integrators = surface_integrators
        self.medium_integrators = medium_integrators
        self.path_state_dim = path_state_dim
        self.surfel_parameters = surfel_parameters
        self.surfel_dim = surfel_dim



class Scene:
    def __init__(self, *visuals: Visual, environment: _core.MapLike = None):
        self.visuals = visuals
        self.geometries = [v.geometry for v in visuals]
        self.transforms = [v.transform for v in visuals]
        self.environment = _core.as_map(environment)

    @cached_property
    def number_of_patches(self):
        return len(self.visuals)

    @cached_property
    def surfaces(self):
        """
        Gets surfaces map for the entire scene with transformed geometries.
        """
        if len(self.geometries) == 0:
            return None
        if len(self.geometries) == 1:
            return self.visuals[0].geometry.surface_map(self.transforms[0])
        return SurfacesMap(self.geometries, self.transforms)

    def surfels(self, surfel_parameters: str):
        return SurfelsMap(self.geometries, self.transforms, surfel_parameters)

    def update(self, just_update=False, reuse=True, **deferred_parameters: _torch.Tensor):
        if isinstance(self.surfaces, SurfacesMap):
            self.surfaces.build_ads(just_update=just_update, reuse=reuse, **deferred_parameters)

    @cached_property
    def geometry_surfaces(self):
        """
        Gets the array of individual geometry surfaces maps in object space (before transformation).
        """
        return [
            SurfacesMap([v.geometry], [_vk.mat4x3.trs()]) for v in self.visuals
        ]

    def __expand_argument_to_list(self, a):
        if a is None:
            return [None] * len(self.visuals)
        if not isinstance(a, list):
            return [a] * len(self.visuals)
        return a

    def __index_list(self, l):
            s = list(set(v for v in l if v))
            return s if s else None, [s.index(v) if v else -1 for v in l]

    def build_transmittance(self, builder: _typing.Callable[
        ['Scene'], _core.MapLike | _typing.List[_core.MapLike]
    ]):
        medium_transmittance = builder(self)
        has_surface = [1 if v.material is not None else 0 for v in self.visuals]
        medium_transmittance = self.__expand_argument_to_list(medium_transmittance)
        medium_transmittance, inside_medium = self.__index_list(medium_transmittance)
        outside_medium = []
        for i, v in enumerate(self.visuals):
            if v.parent is not None and v.parent.medium is not None:
                outside_medium.append(inside_medium[self.visuals.index(v.parent)] if v.parent in self.visuals else -1)
            else:
                outside_medium.append(-1)
        return SceneTransmittance(
            number_of_patches=self.number_of_patches,
            surfaces=self.surfaces,
            has_surface=has_surface,
            outside_medium=outside_medium,
            inside_medium=inside_medium,   # to be filled by integrator
            medium_transmittance=medium_transmittance,
        )

    def build_integrator(self,
                         builder: _typing.Callable[['Scene', int], ScenePathIntegratorObjects],
                         spectral_dim: int)->ScenePathIntegrator:
        objects = builder(self, spectral_dim)
        surface_integrators, medium_integrators, path_state_dim, surfel_parameters, surfel_dim = (
            objects.surface_integrators,
            objects.medium_integrators,
            objects.path_state_dim,
            objects.surfel_parameters,
            objects.surfel_dim
        )
        surface_integrators = self.__expand_argument_to_list(surface_integrators)
        medium_integrators = self.__expand_argument_to_list(medium_integrators)
        surface_integrators, surface_integrator_indices = self.__index_list(surface_integrators)
        medium_integrators, inside_medium = self.__index_list(medium_integrators)  # index pairs
        outside_medium = []
        for i, v in enumerate(self.visuals):
            if v.parent is not None and v.parent.medium is not None:
                outside_medium.append(inside_medium[self.visuals.index(v.parent)] if v.parent in self.visuals else -1)
            else:
                outside_medium.append(-1)
        if isinstance(self.surfaces, SurfacesMap) and not self.surfaces.initialized:
            print("[WARNING] Scene surfaces ADS not built before building integrator. You must update scene before usage, otherwise ADS wont be present")
        return ScenePathIntegrator(
            number_of_patches=self.number_of_patches,
            surfel_dim=surfel_dim,
            spectral_dim=spectral_dim,
            path_state_dim=path_state_dim,
            surfaces=self.surfaces,
            surfels=self.surfels(surfel_parameters=surfel_parameters),
            environment=self.environment,
            surface_integrators=surface_integrators,
            medium_integrators=medium_integrators,
            surface_integrator_indices=surface_integrator_indices,
            outside_medium=outside_medium,
            inside_medium=inside_medium
        )


#
#
# class BoxGeometryDescription(GeometryDescription):
#     def __init__(self, bmin: _vk.vec3 = _vk.vec3(-1.0), bmax: _vk.vec3 = _vk.vec3(1.0)):
#         super().__init__(bmin, bmax)
#
# # class MeshGeometryDescription(GeometryDescription):
# #     pass
#
# class NoMaterialDescription(MaterialDescription):
#     pass
#
#
# class DiffuseMaterialDescription(MaterialDescription):
#     def __init__(self, albedo: _core.TensorLike | _core.deferred):
#         self.albedo = _core.ensure_tensor(albedo, 3)
#
#
# # class SpecularMaterialDescription(MaterialDescription):
# #     def __init__(self, albedo: _core.TensorLike, specular_power: float):
# #         self.albedo = _core.as_map(albedo)
# #         self.specular_power = specular_power
# #
# #
# class FresnelMaterialDescription(MaterialDescription):
#     def __init__(self, reflective_index: float):
#         self.reflective_index = reflective_index
# #
# #
# # class MaterialGroupDescription (MaterialDescription):
# #     def __init__(self, materials: list[MaterialDescription], ratios: list[float]):
# #         self.materials = materials
# #         self.ratios = ratios
#
#
# # =========== Scene Maps =============
#
# def media_integrator(medium: MediumDescription):
#
# class SceneMapBuilder:
#     def __init__(self,
#                  scene_description: SceneDescription
#                  ):
#         self.scene_description = scene_description
#         media = []
#         for visual in scene_description.visuals:
#             if visual.medium is not None:
#                 media.append(visual.medium)
#         if scene_description.global_medium is not None:
#             media.append(scene_description.global_medium)
#         density_type = type(media[0].density) if len(media) > 0 else None
#         scattering_media = list(filter(lambda m: m.scattering_kernel is not None, media))
#         scattering_type = type(scattering_media[0]) if len(scattering_media) > 0 else None
#         assert all(type(m.density) == density_type for m in media), "All media density type in the scene must be of the same type"
#         assert all(m.scattering_kernel is None or type(m.scattering_kernel) == scattering_type for m in media), "All media scattering kernel type in the scene must be of the same type"
#         self.media = media
#         self.density_type = density_type
#         self.scattering_type = scattering_type
#         self.media_integrators = []
#
#
#
#
#     def set_builder(self, key: str, builder: _typing.Callable[[SceneDescription], _core.Map]):
#         """
#         Builds a transmittance map for the scene using raymarching.
#         """
#         raise NotImplementedError()
#
#
# # ========== Coordinate Transforms ==========
#
# class PositionTransformST(_core.Map):
#     __extension_info__ = dict(
#         path=_core.__INCLUDE_PATH__+"/rendering/position_st_transform.h",
#         parameters=dict(
#             scale=_vk.vec3,
#             offset=_vk.vec3
#         )
#     )
#     def __init__(self, scale: _vk.vec3, offset: _vk.vec3, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
#         if input_dim is None:
#             input_dim = 3
#         if output_dim is None:
#             output_dim = 3
#         super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
#         self.scale = scale
#         self.offset = offset
#
#     def clone(self,
#               **kwargs) -> _core.Map:
#         return PositionTransformST(
#             self.scale,
#             self.offset,
#             **kwargs)
#
#
# class RayTransformST(_core.Map):
#     __extension_info__ = dict(
#         path=_core.__INCLUDE_PATH__+"/rendering/ray_st_transform.h",
#         parameters=dict(
#             scale=_vk.vec3,
#             offset=_vk.vec3
#         )
#     )
#
#     def __init__(self, scale: _vk.vec3, offset: _vk.vec3, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
#         if input_dim is None:
#             input_dim = 6
#         if output_dim is None:
#             output_dim = 6
#         super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
#         self.scale = scale
#         self.offset = offset
#
#     def clone(self,
#                 **kwargs) -> _core.Map:
#         return RayTransformST(
#             self.scale,
#             self.offset,
#             **kwargs)
#
#
# class RaySegmentTransformST(_core.Map):
#     __extension_info__ = dict(
#         path=_core.__INCLUDE_PATH__+"/rendering/ray_segment_st_transform.h",
#         parameters=dict(
#             scale=_vk.vec3,
#             offset=_vk.vec3
#         )
#     )
#
#     def __init__(self, scale: _vk.vec3, offset: _vk.vec3, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
#         if input_dim is None:
#             input_dim = 8
#         if output_dim is None:
#             output_dim = 8
#         super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
#         self.scale = scale
#         self.offset = offset
#
#     def clone(self,
#               **kwargs) -> _core.Map:
#         return RaySegmentTransformST(
#             self.scale,
#             self.offset,
#             **kwargs)
#
#
# # ========== Geometry ==========
#
# class Geometry:
#     """
#     Base class for scene geometry defining boundary, surface, surface sampler and occupancy fields.
#     Geometries define a local coordinate system mapping world coordinates from bmin, bmax to unitary volume [-1,1]^3.
#     """
#     def __init__(self, bmin: _vk.vec3 = _vk.vec3(-1.0), bmax: _vk.vec3 = _vk.vec3(1.0)):
#         self.bmin = bmin
#         self.bmax = bmax
#
#     def point_to_volume(self) -> _core.Map:
#         """
#         Returns a Map that transforms points from world coordinates to local geometry coordinates.
#         """
#         scale = _vk.vec3(2.0) / (self.bmax - self.bmin)
#         offset = - (self.bmin + self.bmax) / 2.0 * scale
#         return PositionTransformST(scale=scale, offset=offset)
#
#     def ray_to_volume(self) -> _core.Map:
#         """
#         Returns a Map that transforms rays from world coordinates to local geometry coordinates.
#         """
#         scale = _vk.vec3(2.0) / (self.bmax - self.bmin)
#         offset = - (self.bmin + self.bmax) / 2.0 * scale
#         return RayTransformST(scale=scale, offset=offset)
#
#     def ray_segment_to_volume(self) -> _core.Map:
#         """
#         Returns a Map that transforms ray segments from world coordinates to local geometry coordinates.
#         """
#         scale = _vk.vec3(2.0) / (self.bmax - self.bmin)
#         offset = - (self.bmin + self.bmax) / 2.0 * scale
#         return RaySegmentTransformST(scale=scale, offset=offset)
#
#     def boundary(self) -> _core.Map:
#         """
#         Returns a Map representing the boundary of the scene object as a ray-dependent distance field.
#         """
#         raise NotImplementedError()
#
#     def surface(self) -> _core.Map:
#         """
#         Returns a Map representing the surface of the scene object producing surfels with ray intersections.
#         """
#         raise NotImplementedError()
#
#     def surface_sampler(self) -> _core.Map:
#         """
#         Returns a Map representing a surface sampler that generates surfels on the surface of the scene object.
#         """
#         raise NotImplementedError()
#
#     def occupancy(self) -> _core.Map:
#         """
#         Returns a Map representing the occupancy field of the scene object.
#         """
#         raise NotImplementedError()
#
#
# class BoxBoundary(_core.Map):
#     __extension_info__ = dict(
#         path=_core.__INCLUDE_PATH__+"/rendering/boundary_aabb.h",
#         parameters=dict(
#             bmin=_vk.vec3,
#             bmax=_vk.vec3,
#         )
#     )
#
#     def __init__(self, bmin=_vk.vec3(-1.0), bmax=_vk.vec3(1.0), input_dim=None, output_dim=None, input_requires_grad=None, bw_uses_output=None):
#         if input_dim is None:
#             input_dim = 6
#         if output_dim is None:
#             output_dim = 1
#         assert input_dim == 6, "BoxBoundary requires input_dim of 6 (3 for position, 3 for direction)"
#         assert output_dim == 1, "BoxBoundary requires output_dim of 1 (distance to boundary) Positive if outside, negative if inside, POSINF if no intersection"
#         super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
#         self.bmin = bmin
#         self.bmax = bmax
#
#     def clone(self,
#               **kwargs) -> _core.Map:
#         return BoxBoundary(
#             bmin=self.bmin,
#             bmax=self.bmax, **kwargs)
#
#
# class BoxGeometry(Geometry):
#     def __init__(self, bmin: _vk.vec3 = _vk.vec3(-1.0), bmax: _vk.vec3 = _vk.vec3(1.0)):
#         super().__init__(bmin, bmax)
#
#     def boundary(self) -> _core.Map:
#         return BoxBoundary(self.bmin, self.bmax)
#
#     def surface(self) -> _core.Map:
#         raise NotImplementedError("AABBGeometry does not implement surface(). Use a MeshGeometry or other geometry type.")
#
#     def surface_sampler(self) -> _core.Map:
#         raise NotImplementedError("AABBGeometry does not implement surface_sampler(). Use a MeshGeometry or other geometry type.")
#
#     def occupancy(self) -> _core.Map:
#         return _core.ONE.domain_range(self.bmin, self.bmax)
#
#
# # ========== Material ==========
#
# class BSDF:
#     def bsdf(self) -> _core.Map:
#         """
#         Given a surfel, incoming ray and outgoing ray, returns the BSDF value.
#         """
#         raise NotImplementedError()
#
#     def bsdf_sampler(self) -> _core.Map:
#         """
#         Given a surfel and incoming ray, returns a sampled outgoing ray, the weighted bsdf value and the corresponding PDF.
#         """
#         raise NotImplementedError()
#
#
# # ========== Medium ==========
#
#
# class ScatteringKernel:
#     def scattering(self) -> _core.Map:
#         """
#         Given position, incoming and outgoing directions, returns the phase function value.
#         """
#         raise NotImplementedError()
#
#     def scattering_sampler(self) -> _core.Map:
#         """
#         Given position and incoming direction, returns a sampled outgoing direction, the weighted phase function value and the corresponding PDF.
#         """
#         raise NotImplementedError()
#
#
# class HGPhaseFunction(_core.Map):
#     __extension_info__ = dict(
#         path=_core.__INCLUDE_PATH__+"/rendering/hg_phase.h",
#         parameters=dict(
#             g_factor=_core.Map,
#             scattering_albedo=_core.Map,
#         )
#     )
#     def __init__(self, g_factor: _core.MapLike, scattering_albedo: _core.MapLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
#         if input_dim is None:
#             input_dim = 9
#         scattering_albedo = _core.as_map(scattering_albedo, default=_core.ONE)
#         if output_dim is None:
#             output_dim = scattering_albedo.output_dim
#         else:
#             scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=False)
#         assert input_dim == 9, "HGPhaseFunction requires input_dim of 9 (3 for position, 3 for direction_in, 3 for direction_out)"
#         assert output_dim == scattering_albedo.output_dim, "HGPhaseFunction requires output_dim to be the same that the scattering albedo"
#         super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
#         self.g_factor = _core.as_map(g_factor).cast(input_dim=3, output_dim=1, input_requires_grad=input_requires_grad, bw_uses_output=False)
#         self.scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=False)
#
#     def clone(self,
#               **kwargs) -> 'Map':
#         return HGPhaseFunction(
#             g_factor=self.g_factor,
#             scattering_albedo=self.scattering_albedo,
#             **kwargs)
#
#
# class HGPhaseFunctionSampler(_core.Map):
#     __extension_info__ = dict(
#         path=_core.__INCLUDE_PATH__+"/rendering/hg_phase_sampler.h",
#         parameters=dict(
#             g_factor=_core.Map,
#             scattering_albedo=_core.Map,
#         ),
#         stochastic=True
#     )
#     def __init__(self, g_factor: _core.MapLike, scattering_albedo: _core.MapLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
#         scattering_albedo = _core.as_map(scattering_albedo, default=_core.ONE)
#         if input_dim is None:
#             input_dim = 6
#         if output_dim is None:
#             output_dim = None if scattering_albedo.is_generic_output else 4 + scattering_albedo.output_dim
#         else:
#             scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=output_dim - 4, input_requires_grad=input_requires_grad, bw_uses_output=False)
#         assert input_dim == 6, "HGPhaseFunctionSampler requires input_dim of 6 (3 for position, 3 for direction_in)"
#         super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
#         self.g_factor = _core.as_map(g_factor).cast(input_dim=3, output_dim=1, input_requires_grad=input_requires_grad, bw_uses_output=False)
#         self.scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=output_dim - 4 if output_dim is not None else None, input_requires_grad=input_requires_grad, bw_uses_output=False)
#
#     def clone(self,
#               **kwargs) -> 'Map':
#         return HGPhaseFunctionSampler(
#             g_factor=self.g_factor,
#             scattering_albedo=self.scattering_albedo,
#             **kwargs)
#
#
# class HGScatteringKernel(ScatteringKernel):
#     def __init__(self, g_factor: _core.MapLike, scattering_albedo: _core.MapLike):
#         self._scattering = HGPhaseFunction(g_factor, scattering_albedo)
#         self._scattering_sampler = HGPhaseFunctionSampler(g_factor, scattering_albedo)
#
#     def scattering(self) -> _core.Map:
#         return self._scattering
#
#     def scattering_sampler(self) -> _core.Map:
#         return self._scattering_sampler
#
#
# class FreeFlight:
#     def transmittance(self) -> _core.Map:
#         """
#         Given a ray segment, returns the transmittance along the segment.
#         """
#         raise NotImplementedError()
#
#     def extinction_coefficient(self) -> _core.Map:
#         """
#         Returns a Map representing the extinction coefficient of the medium.
#         """
#         raise NotImplementedError()
#
#     def sample_free_flight(self) -> _core.Map:
#         """
#         Given a ray segment, samples a free-flight distance along the segment, returning the distance, the transmittance up to that point, and the PDF.
#         """
#         raise NotImplementedError()
#
#     def bounded(self, geometry: Geometry) -> 'FreeFlight':
#         """
#         Returns a new FreeFlight instance that is bounded by the given geometry.
#         """
#         return BoundedFreeFlight(self, geometry)
#
#     @classmethod
#     def empty(cls) -> 'FreeFlight':
#         """
#         Returns an empty FreeFlight instance representing a vacuum medium.
#         """
#         return EmptyFreeFlight()
#
#
# class EmptyFreeFlight(FreeFlight):
#     def transmittance(self) -> _core.Map:
#         return _core.ONE
#
#     def extinction_coefficient(self) -> _core.Map:
#         return _core.ZERO
#
#     def sample_free_flight(self) -> _core.Map:
#         return _core.X[:, 6]  # return the full distance along the ray segment
#
#
class RaymarchingIntegrator(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/medium_integrator_rm.h",
        parameters=dict(
            density=_core.Map,
            transform=_torch.Tensor,
            step_size=float,
        )
    )

    def __init__(self,
                 density: _core.MapLike,
                 transform: _core.TensorLike | _core.deferred = None,
                 step_size: float = 0.005,
                 input_dim: int = None,
                 output_dim: int = None,
                 input_requires_grad: bool = False,
                 bw_uses_output: bool = False):
        if transform is None:
            transform = _vk.mat4x3.trs()
        transform = _core.ensure_tensor(transform, 2)
        if input_dim is None:
            input_dim = 7
        spectral_dim = None if output_dim is None else (output_dim - 6) // 2
        assert input_dim == 7, "Transmittances requires 3 for x, 3 for w, 1 for distance"
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
                         SPECTRAL_DIM=spectral_dim)
        self.density = _core.as_map(density).cast(input_requires_grad=False, bw_uses_output=False)
        self.transform = transform
        self.step_size = step_size

    def clone(self,
              **kwargs) -> _core.Map:
        return RaymarchingIntegrator(
            density=self.density,
            transform=self.transform,
            step_size=self.step_size,
            **kwargs)


class DeltatrackingIntegrator(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/medium_integrator_dt.h",
        parameters=dict(
            density=_core.Map,
            majorant=_core.Map,
            transform=_torch.Tensor,
            interaction_integrator=_core.Map,
        ),
        stochastic=True
    )

    def __init__(self,
                 density: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike = None,
                 phase_sampler: _core.MapLike = None,
                 emission: _core.MapLike = None,
                 transform: _core.TensorLike | _core.deferred = None,
                 interaction_integrator: _core.MapLike = None,
                 input_dim: int = None, output_dim: int = None, input_requires_grad: bool = False, bw_uses_output: bool = False):
        if transform is None:
            transform = _vk.mat4x3.trs()
        transform = _core.ensure_tensor(transform, 2)
        density = _core.as_map(density, default=_core.ZERO)
        majorant = _core.as_map(majorant, default=_core.ZERO.cast(output_dim=1) | _core.POSINF.cast(output_dim=1))
        interaction_integrator = _core.as_map(interaction_integrator, default=_core.ZERO)
        if input_dim is None:
            input_dim = 7
        spectral_dim = None if output_dim is None else (output_dim - 6)//2
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
                         SPECTRAL_DIM=spectral_dim)
        self.density = density.cast(input_dim=3, output_dim=1, input_requires_grad=False, bw_uses_output=False)
        self.transform = transform
        self.majorant = majorant.cast(input_dim=6, output_dim=2, input_requires_grad=False, bw_uses_output=False)
        self.interaction_integrator = interaction_integrator.cast(input_dim=6, output_dim=output_dim, input_requires_grad=False, bw_uses_output=False)

    def clone(self,
              **kwargs) -> _core.Map:
        return DeltatrackingIntegrator(
            density=self.density,
            majorant=self.majorant,
            transform=self.transform,
            interaction_integrator=self.interaction_integrator,
            **kwargs)


class RatiotrackingMediumTransmittance(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/medium_transmittance_rt.h",
        parameters=dict(
            density=_core.Map,
            majorant=_core.Map,
            transform=_torch.Tensor,
        ),
        stochastic=True
    )

    def __init__(self, density: _core.MapLike, majorant: _core.MapLike, transform: _core.TensorLike | _core.deferred = None, input_dim: int = None, output_dim: int = None, input_requires_grad: bool = False, bw_uses_output: bool = False):
        if transform is None:
            transform = _vk.mat4x3.trs()
        transform = _core.ensure_tensor(transform, 2)
        density = _core.as_map(density, default=_core.ZERO)
        majorant = _core.as_map(majorant, default=_core.ZERO.cast(output_dim=1) | _core.POSINF.cast(output_dim=1))
        if input_dim is None:
            input_dim = 7
        if output_dim is None:
            output_dim = 1
        assert input_dim == 7, "medium transmittance requires 3 for x, 3 for w, 1 for distance"
        assert output_dim == 1, "medium transmittance requires output_dim of 1 (transmittance)"

        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.density = density.cast(input_dim=3, output_dim=1, input_requires_grad=False, bw_uses_output=False)
        self.transform = transform
        self.majorant = majorant.cast(input_dim=6, output_dim=2, input_requires_grad=False, bw_uses_output=False)

    def clone(self,
              **kwargs) -> 'Map':
        return RatiotrackingMediumTransmittance(
            density=self.density,
            majorant=self.majorant,
            transform=self.transform,
            **kwargs)


class HGPhaseFunction(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/hg_phase.h",
        parameters=dict(
            g_factor=_core.Map,
        )
    )
    def __init__(self, g_factor: _core.MapLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 9
        if output_dim is None:
            output_dim = 1
        assert input_dim == 9, "HGPhaseFunction requires input_dim of 9 (3 for position, 3 for direction_in, 3 for direction_out)"
        assert output_dim == 1, "HGPhaseFunction requires output_dim to be 1"
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.g_factor = _core.as_map(g_factor).cast(input_dim=3, output_dim=1, input_requires_grad=input_requires_grad, bw_uses_output=False)

    def clone(self,
              **kwargs) -> 'Map':
        return HGPhaseFunction(
            g_factor=self.g_factor,
            **kwargs)


class HGPhaseFunctionSampler(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/hg_phase_sampler.h",
        parameters=dict(
            g_factor=_core.Map,
        ),
        stochastic=True
    )
    def __init__(self, g_factor: _core.MapLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 6
        if output_dim is None:
            output_dim = 5
        assert input_dim == 6, "HGPhaseFunctionSampler requires input_dim of 6 (3 for position, 3 for direction_in)"
        assert output_dim == 5, "HGPhaseFunctionSampler requires output_dim to be 5 (3 for sampled direction_out, 1 for sample weight, 1 for 1/PDF)"
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.g_factor = _core.as_map(g_factor).cast(input_dim=3, output_dim=1, input_requires_grad=input_requires_grad, bw_uses_output=False)

    def clone(self,
              **kwargs) -> 'Map':
        return HGPhaseFunctionSampler(
            g_factor=self.g_factor,
            **kwargs)


class SeparableScatteringKernel(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/separable_scattering_kernel.h",
        parameters=dict(
            scattering_albedo=_core.Map,
            phase_function_sampler=_core.Map,
        )
    )

    def __init__(self, scattering_albedo: _core.MapLike, phase_function_sampler: _core.MapLike,
                    input_dim: int = None,
                    output_dim: int = None,
                    input_requires_grad: bool = False,
                    bw_uses_output: bool = False):
        scattering_albedo = _core.as_map(scattering_albedo, default=_core.ZERO)
        phase_function_sampler = _core.as_map(phase_function_sampler, default=_core.ZERO)
        if input_dim is None:
            input_dim = 6
        if output_dim is None:
            output_dim = None if scattering_albedo.is_generic_output else scattering_albedo.output_dim + 6
        spectral_dim = None if output_dim is None else output_dim - 6
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
                         SPECTRAL_DIM=spectral_dim
        )
        self.scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim, input_requires_grad=input_requires_grad, bw_uses_output=False)
        self.phase_function_sampler = phase_function_sampler.cast(input_dim=6, output_dim=5, input_requires_grad=input_requires_grad, bw_uses_output=False)

    def clone(self,
                **kwargs) -> 'Map':
            return SeparableScatteringKernel(
                scattering_albedo=self.scattering_albedo,
                phase_function_sampler=self.phase_function_sampler,
                **kwargs)


class FresnelSurfaceTransport(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/surface_transport_fresnel.h",
        parameters=dict(
            refraction_indices=_torch.Tensor,
            is_single=int,
        ),
        stochastic=True
    )

    def __init__(self, refraction_indices: _core.TensorLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        refraction_indices = _core.as_tensor(refraction_indices)
        if output_dim is None and refraction_indices.shape[0] > 1:
            output_dim = refraction_indices.shape[0] + 6
        if input_dim is None:
            input_dim = 14  # 3 incomming direction, 3 position, 3 geometric normal, 3 shading normal, 2 uv
        assert input_dim == 14, "FresnelSurfaceIntegrator requires input_dim of 9 (3 for position, 3 for normal, 3 for incoming direction)"
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM = None if output_dim is None else output_dim - 6
        )
        self.refraction_indices = refraction_indices
        self.is_single = 1 if refraction_indices.shape[0] == 1 else 0

    def clone(self,
              **kwargs) -> 'Map':
        return FresnelSurfaceTransport(
            refraction_indices=self.refraction_indices,
            **kwargs)


class DiffuseSurfaceTransport(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/surface_transport_diffuse.h",
        parameters=dict(
            albedo=_core.Map,
        ),
        stochastic=True
    )

    def __init__(self, albedo: _core.MapLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        albedo = _core.as_map(albedo, default=_core.ONE)
        if input_dim is None:
            input_dim = 14  # 3 incomming direction, 3 position, 3 geometric normal, 3 shading normal, 2 uv
        assert input_dim == 14, "DiffuseSurfaceIntegrator requires input_dim of 9 (3 for position, 3 for normal, 3 for incoming direction)"
        if output_dim is None:
            output_dim = albedo.output_dim + 6
        super().__init__(
            input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM = None if output_dim is None else output_dim - 6
        )
        self.albedo = albedo.cast(input_dim=2, output_dim=output_dim-6 if output_dim is not None else None, input_requires_grad=input_requires_grad, bw_uses_output=False)

    def clone(self,
              **kwargs) -> 'Map':
        return DiffuseSurfaceTransport(
            self.albedo,
            **kwargs)


class BSDFSurfaceIntegrator(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/surface_integrator_bsdf.h",
        parameters=dict(
            bsdf_sampler=_core.Map,
            emission=_core.Map
        )
    )

    required_surfel_parameters = "NT"

    def __init__(self,
                 bsdf_sampler: _core.MapLike,
                 emission: _core.MapLike,
                 surfel_parameters: str,
                 spectral_dim: int,
                 path_state_dim: int,
                 input_dim: int = None,
                 output_dim: int = None,
                 input_requires_grad: bool = False,
                 bw_uses_output: bool = False):
        """
        input_dim = 6 + surfel_dim
        output_dim = 6 + spectral_dim*2
        bsdf_sampler input_dim = 3 + surfel_dim
        bsdf_sampler output_dim = 3 + 1 + spectral_dim
        """
        surfel_dim, surfel_indices = compute_surfel_indices(surfel_parameters)
        bsdf_sampler = _core.as_map(bsdf_sampler, default=_core.ZERO)
        emission = _core.as_map(emission, default=_core.ZERO)
        if input_dim is None:
            input_dim = 6 + surfel_dim
        if output_dim is None:
            output_dim = 6 + spectral_dim * 2
        assert input_dim == 6 + surfel_dim + path_state_dim, "BSDFSurfaceIntegrator requires input_dim of 6 + surfel_dim"
        assert output_dim == 6 + spectral_dim * 2 + path_state_dim, "BSDFSurfaceIntegrator requires output_dim of 6 + spectral_dim*2"
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
                            SURFEL_DIM=surfel_dim,
                            **surfel_indices,
                            SPECTRAL_DIM=spectral_dim,
                            PATH_STATE_DIM=path_state_dim
                         )
        self.bsdf_sampler = bsdf_sampler.cast(input_dim=3 + surfel_dim, output_dim=3 + 1 + spectral_dim, input_requires_grad=False, bw_uses_output=False)
        self.emission = emission.cast(input_dim=3 + surfel_dim, output_dim=spectral_dim, input_requires_grad=False, bw_uses_output=False)
        self.surfel_parameters = surfel_parameters

    def clone(self,
              **kwargs) -> _core.Map:
        return BSDFSurfaceIntegrator(
            self.bsdf_sampler,
            self.emission,
            self.surfel_parameters,
            spectral_dim=self.rdv_generics["SPECTRAL_DIM"],
            path_state_dim=self.rdv_generics["PATH_STATE_DIM"],
            **kwargs)


def material_integrator_bsdf(s: Scene, spectral_dim: int, path_state_dim: int):
    """
    Collects all basic materials and builds a mixture sampler for type for the scene.
    Return a list of BSDFSurfaceIntegrator instances for each visual in the scene, surfel_parameters and surfel_dim
    """

    # collect required surfel parameters
    surfel_parameters_set = set()
    for v in s.visuals:
        if v.material is not None:
            surfel_parameters_set = surfel_parameters_set.union(set(v.material.required_surfel_parameters()))
    # include surfel parameters required by the surface integrator
    surfel_parameters_set.update(BSDFSurfaceIntegrator.required_surfel_parameters)
    surfel_parameters = str(surfel_parameters_set)
    surfel_dim, _ = compute_surfel_indices(surfel_parameters)

    maps = []
    masks = []
    # collect all material bsdf samplers maps
    for v in s.visuals:
        if v.material is None:
            maps.append(None)
            masks.append(None)
            continue
        assert v.material is not None
        # TODO: handle layered instead
        sub_mats = v.material.flatten_lobes(surfel_parameters)
        mix_maps = [None if s is None else s.bsdf_sampler() for _, s in sub_mats[0]]
        mix_masks = [[m.mask_to_sample(surfel_parameters) for m in mask_list] for mask_list, _ in sub_mats[0]]
        maps.append(mix_maps)
        masks.append(mix_masks)
    # cast all maps to the same spectral dimension and surfel_dim
    maps = [[m.cast(input_dim=3 + surfel_dim, output_dim=spectral_dim + 3 + 1) if m is not None else None for m in
             mix_maps] if mix_maps is not None else None for mix_maps in maps]
    masks = [[[m.cast(input_dim=3 + surfel_dim, output_dim=1) for m in mask_list] for mask_list in
              mix_masks] if mix_masks is not None else None for mix_masks in masks]
    # compute general mask_id
    mask_id = 0
    max_masks = 0
    for mix_masks in masks:
        if mix_masks is not None:
            for mask_list in mix_masks:
                max_masks = max(max_masks, len(mask_list))
                for m in mask_list:
                    if mask_id == 0:
                        mask_id = m.rdv_kernel_id
                    else:
                        assert m.rdv_kernel_id == mask_id, "All masks must share the same rdv_kernel_id."
    max_repeats = 0
    for mix_maps in maps:
        repeats = {}
        if mix_maps is not None:
            for m in mix_maps:
                if m is not None:
                    if m.rdv_kernel_id not in repeats:
                        repeats[m.rdv_kernel_id] = 0
                    repeats[m.rdv_kernel_id] += 1
            max_repeats = max(max_repeats, max(repeats.values()) if len(repeats) > 0 else 0)
    ids = set(
        m.rdv_kernel_id for repeated_maps in maps if repeated_maps is not None for m in repeated_maps if m is not None)
    type = create_mixture_sampler(list(ids), mask_id, max_repeats, max_masks)
    return surfel_dim, surfel_parameters, [BSDFSurfaceIntegrator(type(mix_maps, mix_masks), emission=None, surfel_parameters=surfel_parameters, spectral_dim=spectral_dim, path_state_dim=path_state_dim) if mix_maps is not None else None for mix_maps, mix_masks in zip(maps, masks)]


class MediumIntegratorSingleFF(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/medium_integrator_single_ff.h",
        parameters=dict(
            event_sampler=_core.Map,
            scattering_sampler=_core.Map,
        )
    )
    def __init__(self, event_sampler: _core.MapLike, scattering_sampler: _core.MapLike = None, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        event_sampler = _core.as_map(event_sampler)
        scattering_sampler = _core.as_map(scattering_sampler, default=_core.ZERO)
        spectral_dim = None
        if output_dim is not None:
            spectral_dim = (output_dim - 6)//2
        if not scattering_sampler.is_generic_output:
            spectral_dim = scattering_sampler.output_dim - 4
        if spectral_dim is not None:
            if output_dim is not None:
                assert output_dim == 6 + spectral_dim * 2, "Output dim must be 6 + 2*spectral_dim"
            else:
                output_dim = 6 + spectral_dim * 2
            scattering_sampler = scattering_sampler.cast(input_dim=6 + event_sampler.output_dim - 1, output_dim=4 + spectral_dim, input_requires_grad=input_requires_grad, bw_uses_output=False)
        event_sampler = event_sampler.cast(input_dim=7)
        if input_dim is None:
            input_dim = 7
        assert input_dim == 7
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
                         SPECTRAL_DIM=spectral_dim,
                         EVENT_INFO_DIM=event_sampler.output_dim - 1
                         )
        self.event_sampler = event_sampler
        self.scattering_sampler = scattering_sampler

    def clone(self,
              **kwargs) -> _core.Map:
        return MediumIntegratorSingleFF(
            self.event_sampler,
            self.scattering_sampler,
            **kwargs)


class MediumIntegratorDT(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__+"/rendering/medium_integrator_dt.h",
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            transform=_torch.Tensor,
        ),
        stochastic=True
    )

    def __init__(self, extinction: _core.MapLike, majorant: _core.MapLike, scattering_albedo: _core.MapLike, anisotropy: _core.MapLike, transform: _core.TensorLike | _core.deferred = None, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if transform is None:
            transform = _vk.mat4x3.trs()
        transform = _core.ensure_tensor(transform, 2)
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        spectral_dim = None
        if output_dim is not None:
            spectral_dim = (output_dim - 6)//2
        if spectral_dim is not None:
            if output_dim is not None:
                assert output_dim == 6 + spectral_dim * 2, "Output dim must be 6 + 2*spectral_dim"
            else:
                output_dim = 6 + spectral_dim * 2
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim, input_requires_grad=input_requires_grad, bw_uses_output=False)
        anisotropy = anisotropy.cast(input_dim=3, output_dim=1, input_requires_grad=False, bw_uses_output=False)
        extinction = extinction.cast(input_dim=3, output_dim=1, input_requires_grad=False, bw_uses_output=False)
        majorant = majorant.cast(input_dim=6, output_dim=2, input_requires_grad=False, bw_uses_output=False)
        if input_dim is None:
            input_dim = 7
        if spectral_dim is not None:
            assert output_dim is None or output_dim == 6 + spectral_dim * 2
            output_dim = 6 + spectral_dim * 2
        assert input_dim == 7
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
                         SPECTRAL_DIM=spectral_dim
                         )
        self.extinction = extinction
        self.majorant = majorant
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return MediumIntegratorDT(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            transform=self.transform,
            **kwargs)


def medium_integrator_single(s: Scene):
    medium_integrators = []
    for v in s.visuals:
        if v.medium is not None:
            model = v.medium.scatterers()[0]
            event_sampler = model.event_sampler()
            scattering_sampler = model.scattering_sampler()
            medium_integrators.append(
                MediumIntegratorSingleFF(event_sampler, scattering_sampler)
            )
        else:
            medium_integrators.append(None)
    return medium_integrators


def medium_integrator_dt(s: Scene):
    medium_integrators = []
    for v in s.visuals:
        if v.medium is not None:
            scatterer = v.medium.scatterers()[0]
            assert isinstance(scatterer, HGMediumScatterer), "For DeltatrackingIntegrator, the medium scatterer must be an HGMediumScatterer."
            extinction = scatterer.extinction
            majorant = scatterer.majorant
            scattering_albedo = scatterer.scattering_albedo
            anisotropy = scatterer.anisotropy
            transform = scatterer.transform
            medium_integrators.append(
                MediumIntegratorDT(extinction, majorant, scattering_albedo, anisotropy, transform)
            )
        else:
            medium_integrators.append(None)
    return medium_integrators


def medium_transmittance_rt(s: Scene):
    medium_transmittance = []
    for v in s.visuals:
        if v.medium is not None:
            extinction_map = v.medium.density.get_exitinction_map()
            majorant_map = v.medium.density.get_majorant_map()
            medium_transmittance.append(
                RatiotrackingMediumTransmittance(
                    density=extinction_map,
                    majorant=majorant_map,
                    transform=v.transform
                )
            )
        else:
            medium_transmittance.append(None)
    return medium_transmittance


#
#
# class RaymarchingFreeFlight(FreeFlight):
#     def __init__(self, extinction_coefficient: _core.Map, step_size: float = 0.005):
#         self._extinction_coefficient = extinction_coefficient
#         self.step_size = step_size
#         self._transmittance_cache = None
#         self._free_flight_sampler = None
#
#     def extinction_coefficient(self) -> _core.Map:
#         return self._extinction_coefficient
#
#     def transmittance(self) -> _core.Map:
#         if self._transmittance_cache is None:
#             self._transmittance_cache = RaymarchingTransmittance(
#                 density=self._extinction_coefficient,
#                 step_size=self.step_size
#             )
#         return self._transmittance_cache
#
#
# class DeltatrackingTransmittance(_core.Map):
#     __extension_info__ = dict (
#         path=_core.__INCLUDE_PATH__+"/rendering/transmittance_dt.h",
#         parameters=dict(
#             density=_core.Map,
#             majorant=_core.Map,
#         ),
#         stochastic=True
#     )
#
#     def __init__(self, density: _core.MapLike, majorant: _core.MapLike, input_dim: int = None, output_dim: int = None, input_requires_grad: bool = False, bw_uses_output: bool = False):
#         density = _core.as_map(density, default=_core.ZERO)
#         majorant = _core.as_map(majorant, default=_core.ZERO.cast(output_dim=1) | _core.POSINF.cast(output_dim=1))
#         if input_dim is None:
#             input_dim = 8
#         if output_dim is None:
#             output_dim = 1
#         super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
#         self.density = density.cast(input_dim=3, output_dim=1, input_requires_grad=input_requires_grad, bw_uses_output=False)
#         self.majorant = majorant.cast(input_dim=6, output_dim=2, input_requires_grad=False, bw_uses_output=False)
#
#     def clone(self,
#               **kwargs) -> 'Map':
#         return DeltatrackingTransmittance(
#             density=self.density,
#             majorant=self.majorant,
#             **kwargs)
#
#
# class DeltatrackingFreeFlightSampler(_core.Map):
#     __extension_info__ = dict (
#         path=_core.__INCLUDE_PATH__+"/rendering/free_flight_sampler_dt.h",
#         parameters=dict(
#             density=_core.Map,
#             majorant=_core.Map,
#         ),
#         stochastic=True
#     )
#
#     def __init__(self, density: _core.MapLike, majorant: _core.MapLike, input_dim: int = None, output_dim: int = None, input_requires_grad: bool = False, bw_uses_output: bool = False):
#         density = _core.as_map(density, default=_core.ZERO)
#         majorant = _core.as_map(majorant, default=_core.ZERO.cast(output_dim=1) | _core.POSINF.cast(output_dim=1))
#         if input_dim is None:
#             input_dim = 8
#         if output_dim is None:
#             output_dim = 1
#         super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
#         self.density = density.cast(input_dim=3, output_dim=1, input_requires_grad=input_requires_grad, bw_uses_output=False)
#         self.majorant = majorant.cast(input_dim=6, output_dim=2, input_requires_grad=False, bw_uses_output=False)
#
#     def clone(self,
#               **kwargs) -> 'Map':
#         return DeltatrackingFreeFlightSampler(
#             density=self.density,
#             majorant=self.majorant,
#             **kwargs)
#
#
# class DeltatrackingFreeFlight(FreeFlight):
#     def __init__(self,
#                  extinction_coefficient: _core.MapLike,
#                  max_extinction: float
#         ):
#         self._extinction_coefficient = _core.as_map(extinction_coefficient, default=_core.ZERO)
#         self._max_extinction_tensor = _vk.tensor_copy(_torch.tensor([max_extinction, 100000.0]))
#         self._majorant = _core.ConstantMap(self._max_extinction_tensor)
#         self._transmittance_cache = None
#         self._free_flight_sampler_cache = None
#
#     def update_majorant(self, value: float):
#         self._majorant[0] = value
#
#     def extinction_coefficient(self) -> _core.Map:
#         return self._extinction_coefficient
#
#     def transmittance(self) -> _core.Map:
#         if self._transmittance_cache is None:
#             self._transmittance_cache = DeltatrackingTransmittance(
#                 self._extinction_coefficient,
#                 self._majorant
#             )
#         return self._transmittance_cache
#
#     def sample_free_flight(self) -> _core.Map:
#         if self._free_flight_sampler_cache is None:
#             self._free_flight_sampler_cache = DeltatrackingFreeFlightSampler(
#                 self._extinction_coefficient,
#                 self._majorant
#             )
#         return self._free_flight_sampler_cache
#
#
# class BoundedTransmittance(_core.Map):
#     __extension_info__ = dict (
#         path=_core.__INCLUDE_PATH__+"/rendering/bounded_transmittance.h",
#         parameters=dict(
#             base_transmittance=_core.Map,
#             boundary=_core.Map,
#             bmin=_vk.vec3,
#             bmax=_vk.vec3,
#         )
#     )
#
#     def __init__(self,
#                  base_transmittance: _core.MapLike,
#                  boundary: _core.MapLike, bmin: _vk.vec3 = _vk.vec3(-1.0), bmax: _vk.vec3 = _vk.vec3(1.0),
#                  input_dim: int = None, output_dim: int = None, input_requires_grad: bool = False, bw_uses_output: bool = False):
#         if input_dim is None:
#             input_dim = 8
#         if output_dim is None:
#             output_dim = 1
#         assert input_dim == 8, "Transmittances requires 3 for x, 3 for w, 1 for distance, 1 for density scale"
#         assert output_dim == 1, "Transmittances requires output_dim of 1 (transmittance)"
#         super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
#         self.base_transmittance = _core.as_map(base_transmittance).cast(input_dim=8, output_dim=1, input_requires_grad=input_requires_grad, bw_uses_output=True)
#         self.boundary = _core.as_map(boundary).cast(input_dim=6, output_dim=1, input_requires_grad=False, bw_uses_output=False)
#         self.bmin = bmin
#         self.bmax = bmax
#
#     def clone(self,
#               **kwargs) -> _core.Map:
#         return BoundedTransmittance(
#             base_transmittance=self.base_transmittance,
#             boundary=self.boundary,
#             bmin=self.bmin,
#             bmax=self.bmax,
#             **kwargs)
#
#
# class BoundedFreeFlightSampler(_core.Map):
#     __extension_info__ = dict (
#         path=_core.__INCLUDE_PATH__+"/rendering/bounded_free_flight_sampler.h",
#         parameters=dict(
#             base_free_flight_sampler=_core.Map,
#             boundary=_core.Map,
#             bmin=_vk.vec3,
#             bmax=_vk.vec3,
#         )
#     )
#
#     def __init__(self,
#                  base_free_flight_sampler: _core.MapLike,
#                  boundary: _core.MapLike,
#                  bmin: _vk.vec3 = _vk.vec3(-1.0), bmax: _vk.vec3 = _vk.vec3(1.0),
#                  input_dim: int = None, output_dim: int = None, input_requires_grad: bool = False, bw_uses_output: bool = False):
#         if input_dim is None:
#             input_dim = 8
#         if output_dim is None:
#             output_dim = 1
#         assert input_dim == 8, "Free flight samplers requires 3 for x, 3 for w, 1 for distance, 1 for density scale"
#         assert output_dim == 1, "Free flight samplers requires output_dim of 1 (sampled distance)"
#         super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
#         self.base_free_flight_sampler = _core.as_map(base_free_flight_sampler).cast(input_dim=8, output_dim=1, input_requires_grad=input_requires_grad, bw_uses_output=False)
#         self.boundary = _core.as_map(boundary).cast(input_dim=6, output_dim=1, input_requires_grad=False, bw_uses_output=False)
#         self.bmin = bmin
#         self.bmax = bmax
#
#     def clone(self,
#               **kwargs) -> _core.Map:
#         return BoundedFreeFlightSampler(
#             base_free_flight_sampler=self.base_free_flight_sampler,
#             boundary=self.boundary,
#             bmin=self.bmin,
#             bmax=self.bmax,
#             **kwargs)
#
#
# class BoundedFreeFlight(FreeFlight):
#     def __init__(self, base_free_flight: FreeFlight, geometry: Geometry):
#         self.base_free_flight = base_free_flight
#         self.geometry = geometry
#         self._extinction_coefficient_cache = None
#         self._transmittance_cache = None
#         self._free_flight_sampler_cache = None
#
#     def extinction_coefficient(self) -> _core.Map:
#         if self._extinction_coefficient_cache is None:
#             base_extinction = self.base_free_flight.extinction_coefficient()
#             transformed_extinction = base_extinction.after(self.geometry.point_to_volume())
#             bounded_extinction = transformed_extinction.domain_mask(self.geometry.occupancy())
#             self._extinction_coefficient_cache = bounded_extinction
#         return self._extinction_coefficient_cache
#
#     def transmittance(self) -> _core.Map:
#         if self._transmittance_cache is None:
#             base_transmittance = self.base_free_flight.transmittance()
#             bounded_transmittance = BoundedTransmittance(
#                 base_transmittance,
#                 boundary=self.geometry.boundary(),
#                 bmin=self.geometry.bmin,
#                 bmax=self.geometry.bmax
#             )
#             self._transmittance_cache = bounded_transmittance
#         return self._transmittance_cache
#
#     def sample_free_flight(self) -> _core.Map:
#         if self._free_flight_sampler_cache is None:
#             base_sampler = self.base_free_flight.sample_free_flight()
#             bounded_sampler = BoundedFreeFlightSampler(
#                 base_sampler,
#                 boundary=self.geometry.boundary(),
#                 bmin=self.geometry.bmin,
#                 bmax=self.geometry.bmax
#             )
#             self._free_flight_sampler_cache = bounded_sampler
#         return self._free_flight_sampler_cache
#
#
# class Material:
#     def __init__(self,
#                     bsdf: BSDF,  # surface interaction
#                     emission: _core.Map,  # surface emission
#                  ):
#         self.bsdf = bsdf
#         self.emission = emission
#
#
# class Medium:
#     def __init__(self,
#                     free_flight: FreeFlight,
#                     scattering_kernel: ScatteringKernel | None = None,
#                     volume_emission: _core.MapLike = None
#                     ):
#         self.free_flight = free_flight
#         self.scattering_kernel = scattering_kernel
#         self.volume_emission = volume_emission
#
#
# class Environment:
#     def emission(self) -> _core.Map:
#         """
#         Given a ray direction, returns the environment emission in that direction.
#         """
#         raise NotImplementedError()
#
#     def emission_sampler(self) -> _core.Map:
#         """
#         Samples a direction from the environment, returning the direction, the weighted emission value and the corresponding PDF.
#         """
#         raise NotImplementedError()
#
#
# class Visual:
#     def __init__(self,
#                     geometry: Geometry,
#                     transform: _torch.Tensor | _core.deferred,
#                     material: Material | None = None,
#                     medium: Medium | None = None,
#                     ):
#         self.geometry = geometry
#         self.transform = _core.ensure_tensor(transform, 2)
#         self.material = material
#         self.medium = medium
#
#
# class SceneTransmittance(_core.Map):
#     __extension_info__ = dict(
#         path=_core.__INCLUDE_PATH__+"/rendering/scene_transmittance.h",
#         parameters=dict(
#             medium_transmittance=_core.Map,
#             boundary=_core.Map,
#         )
#     )
#
#     def __init__(self, medium_transmittance: _core.MapLike, boundary: _core.MapLike, input_dim: int = None, output_dim: int = None, input_requires_grad: bool = False, bw_uses_output: bool = False):
#         if input_dim is None:
#             input_dim = 8
#         if output_dim is None:
#             output_dim = 1
#         medium_transmittance = _core.as_map(medium_transmittance, default=_core.ONE)
#         boundary = _core.as_map(boundary, default=_core.POSINF)
#         medium_transmittance = medium_transmittance.cast(
#             input_dim=8,
#             output_dim=1,
#             input_requires_grad=input_requires_grad,
#             bw_uses_output=bw_uses_output
#         )
#         boundary = boundary.cast(
#             input_dim=6,
#             output_dim=1,
#             input_requires_grad=False,
#             bw_uses_output=False
#         )
#         assert input_dim == 8, "SceneTransmittance requires input_dim of 8 (3 for position, 3 for direction, 1 for distance, 1 for density scale)"
#         assert output_dim == 1, "SceneTransmittance requires output_dim of 1 (transmittance)"
#         super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
#         self.medium_transmittance = medium_transmittance
#         self.boundary = boundary
#
#     def clone(self,
#                 **kwargs) -> _core.Map:
#             return SceneTransmittance(
#                 medium_transmittance=self.medium_transmittance,
#                 boundary=self.boundary,
#                 **kwargs)
#
#
# class Scene:
#     def __init__(self,
#                     *visuals: Visual,  # list of visuals in the scene (geometry + properties)
#                     medium: Medium = None,  # medium for the entire scene (represents all volumes between visuals)
#                     environment: Environment = None,  # environment surrounding the scene
#                  ):
#         self.visuals = visuals
#         self.medium = medium
#         self.environment = environment
#         self._transmittance_cache = None
#
#     def transmittance(self) -> _core.Map:
#         """
#         Given a ray segment, returns the transmittance along the segment considering all visuals in the scene.
#         """
#         if self._transmittance_cache is None:
#             self._transmittance_cache = SceneTransmittance(
#                 medium_transmittance=self.medium.free_flight.transmittance() if self.medium is not None else _core.ONE,
#                 boundary=_core.POSINF  # TODO: combine boundaries of all visuals
#             )
#         return self._transmittance_cache
#
#     def direct_radiance(self) -> _core.Map:
#         """
#         Given a ray segment, returns the direct radiance along the segment considering all visuals and the environment in the scene.
#         """
#         raise NotImplementedError()
#
#     def scattered_radiance(self) -> _core.Map:
#         """
#         Given a ray segment, returns the scattered radiance along the segment considering all visuals and the environment in the scene.
#         """
#         raise NotImplementedError()
#
#     def emission(self) -> _core.Map:
#         """
#         Given a ray segment, returns the total emission from the scene in that direction considering all visuals and the environment.
#         """
#         raise NotImplementedError()



