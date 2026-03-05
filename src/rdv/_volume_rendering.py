from . import _core as _core
import vulky as _vk
import torch as _torch


class PBVR(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/pbvr.h',
        parameters = dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            environment=_core.Map,
            environment_sampler=_core.Map,
            # bmin=_vk.vec3,
            # bmax=_vk.vec3,
            transform=_torch.Tensor
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 environment: _core.MapLike = None,
                 environment_sampler: _core.MapLike = None,
                 transform: _core.TensorLike | _core.deferred | None = None,
                 # bmin: _vk.vec3 = _vk.vec3(-1.0, -1.0, -1.0),
                 # bmax: _vk.vec3 = _vk.vec3(1.0, 1.0, 1.0),
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        assert environment is not None or environment_sampler is not None, "Both environment and environment_sampler cannot be None, result would be null"
        if transform is None:
            transform = _vk.mat4x3.trs()
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        ENVIRONMENT = 1 if environment is not None else None
        environment = _core.as_map(environment, default=_core.ZERO)
        ENVIRONMENT_SAMPLER = 1 if environment_sampler is not None else None
        environment_sampler = _core.as_map(environment_sampler, default=_core.ZERO)

        if input_dim is None:
            input_dim = 6
        output_dim = output_dim or scattering_albedo.output_dim
        output_dim = output_dim or environment.output_dim
        if environment_sampler.output_dim is not None:
            output_dim = output_dim or (environment_sampler.output_dim - 4)

        extinction = extinction.cast(input_dim=3, output_dim=1)
        majorant = majorant.cast(input_dim=6, output_dim=2)
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=output_dim)
        anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        environment = environment.cast(input_dim=3, output_dim=output_dim)
        environment_sampler = environment_sampler.cast(input_dim=6, output_dim=output_dim+4)
        transform = _core.ensure_tensor(transform, map_dim=2)

        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            ENVIRONMENT=ENVIRONMENT, ENVIRONMENT_SAMPLER=ENVIRONMENT_SAMPLER, SPECTRAL_DIM=output_dim
        )
        self.extinction = extinction
        self.majorant = majorant
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy
        self.environment = environment
        self.environment_sampler = environment_sampler
        self.transform = transform
        # self.bmin = bmin
        # self.bmax = bmax

    def clone(self,
              **kwargs) -> _core.Map:
        return PBVR(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            environment=self.environment,
            environment_sampler=self.environment_sampler,
            # bmin=self.bmin,
            # bmax=self.bmax,
            transform=self.transform,
            **kwargs
        )


class RatioTrackingTransmittance(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/transmittance_rt.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            transform=_torch.Tensor
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 transform: _core.TensorLike | _core.deferred | None = None,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if transform is None:
            transform = _vk.mat4x3.trs()
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        if input_dim is None:
            input_dim = 6
        output_dim = output_dim or 1
        assert output_dim == 1, "RatioTrackingTransmittance only supports output_dim=1, got output_dim={output_dim}"
        extinction = extinction.cast(input_dim=3, output_dim=1)
        majorant = majorant.cast(input_dim=6, output_dim=2)
        transform = _core.ensure_tensor(transform, map_dim=2)

        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output
        )
        self.extinction = extinction
        self.majorant = majorant
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return RatioTrackingTransmittance(
            extinction=self.extinction,
            majorant=self.majorant,
            transform=self.transform,
            **kwargs
        )


class ScatteringSampler(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/scattering_sampler.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            anisotropy=_core.Map,
            transform=_torch.Tensor
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 anisotropy: _core.MapLike,
                 transform: _core.TensorLike | _core.deferred | None = None,
                 # bmin: _vk.vec3 = _vk.vec3(-1.0, -1.0, -1.0),
                 # bmax: _vk.vec3 = _vk.vec3(1.0, 1.0, 1.0),
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if transform is None:
            transform = _vk.mat4x3.trs()
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        anisotropy = _core.as_map(anisotropy)
        if input_dim is None:
            input_dim = 6
        if output_dim is None:
            output_dim = 4
        assert output_dim == 4
        extinction = extinction.cast(input_dim=3, output_dim=1)
        majorant = majorant.cast(input_dim=6, output_dim=2)
        anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        transform = _core.ensure_tensor(transform, map_dim=2)

        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output
        )
        self.extinction = extinction
        self.majorant = majorant
        self.anisotropy = anisotropy
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ScatteringSampler(
            extinction=self.extinction,
            majorant=self.majorant,
            anisotropy=self.anisotropy,
            transform=self.transform,
            **kwargs
        )


