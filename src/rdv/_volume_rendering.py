from pandas.core.array_algos import transforms

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


class PBVRT(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/pbvrt.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            environment=_core.Map,
            environment_sampler=_core.Map,
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

        spectral_dim = scattering_albedo.output_dim or environment.output_dim or (environment_sampler.output_dim - 4 if environment_sampler.output_dim is not None else None)
        if output_dim is None:
            output_dim = spectral_dim + 1 if spectral_dim is not None else None
        assert output_dim == spectral_dim + 1 if spectral_dim is not None else True, f"Output dim must be spectral dim + 1 (for transmittance), got output_dim={output_dim} and spectral_dim={spectral_dim}"

        extinction = extinction.cast(input_dim=3, output_dim=1)
        majorant = majorant.cast(input_dim=6, output_dim=2)
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim)
        anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        environment = environment.cast(input_dim=3, output_dim=spectral_dim)
        environment_sampler = environment_sampler.cast(input_dim=6, output_dim=spectral_dim + 4)
        transform = _core.ensure_tensor(transform, map_dim=2)

        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            ENVIRONMENT=ENVIRONMENT, ENVIRONMENT_SAMPLER=ENVIRONMENT_SAMPLER, SPECTRAL_DIM=spectral_dim
        )
        self.extinction = extinction
        self.majorant = majorant
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy
        self.environment = environment
        self.environment_sampler = environment_sampler
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return PBVRT(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            environment=self.environment,
            environment_sampler=self.environment_sampler,
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


class RaymarchingTransmittance(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/transmittance_rm.h',
        parameters=dict(
            extinction=_core.Map,
            step_size=float,
            transform=_torch.Tensor
        )
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 step_size: float = 0.005,
                 transform: _core.TensorLike | _core.deferred | None = None,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if transform is None:
            transform = _vk.mat4x3.trs()
        extinction = _core.as_map(extinction)
        if input_dim is None:
            input_dim = 6
        output_dim = output_dim or 1
        assert output_dim == 1, "RaymarchingTransmittance only supports output_dim=1, got output_dim={output_dim}"
        extinction = extinction.cast(input_dim=3, output_dim=1)
        transform = _core.ensure_tensor(transform, map_dim=2)

        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output
        )
        self.extinction = extinction
        self.step_size = step_size
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return RaymarchingTransmittance(
            extinction=self.extinction,
            step_size=self.step_size,
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


class RaymarchingRayIntegral(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/ray_integral_rm.h',
        parameters=dict(
            integrand=_core.Map,
            transform=_torch.Tensor,
            step_size=float,
        ),
        stochastic=True,
    )

    def __init__(self, integrand: _core.MapLike, transform: _torch.Tensor | _vk.mat4x3 | None = None, step_size: float = 0.005, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        integrand = _core.as_map(integrand).cast(input_dim=3)
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        assert input_dim == 6
        if output_dim is None:
            output_dim = integrand.output_dim
        assert output_dim is None or output_dim == integrand.output_dim
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output, SPECTRAL_DIM=output_dim)
        self.integrand = integrand.cast(output_dim=output_dim)
        self.transform = transform
        self.step_size = step_size

    def clone(self, **kwargs) -> _core.Map:
        return RaymarchingRayIntegral(
            integrand=self.integrand,
            transform=self.transform,
            step_size=self.step_size,
            **kwargs
        )


class TransmittanceProfiling(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/transmittance_profiling.h',
        parameters=dict(
            extinction=_core.Map,
            step_size=float,
            transform=_torch.Tensor,
            transmittance_threshold=['OUTPUT_DIM', float]
        )
    )

    def __init__(self, extinction: _core.MapLike, transmittance_threshold: list[float], produce_transmittance: bool = False, step_size: float = 0.005, transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None:
            output_dim = len(transmittance_threshold) + (1 if produce_transmittance else 0)
        assert input_dim == 6
        assert output_dim == len(transmittance_threshold) + (1 if produce_transmittance else 0)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            PRODUCE_TRANSMITTANCE=1 if produce_transmittance else None,
        )
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.transform = transform
        self.step_size = step_size

    def clone(self,
              **kwargs) -> _core.Map:
        return TransmittanceProfiling(
            extinction=self.extinction,
            transmittance_threshold=[self.transmittance_threshold[i] for i in range(self.output_dim)],
            step_size=self.step_size,
            transform=self.transform,
            produce_transmittance='PRODUCE_TRANSMITTANCE' in self.rdv_generics,
            **kwargs
        )


class TransmittanceProfiling2(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/transmittance_profiling_2.h',
        parameters=dict(
            extinction=_core.Map,
            step_size=float,
            transform=_torch.Tensor,
            transmittance_threshold=['OUTPUT_DIM', float]
        )
    )

    def __init__(self, extinction: _core.MapLike, transmittance_threshold: list[float], produce_transmittance: bool = False, step_size: float = 0.005, transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None:
            output_dim = len(transmittance_threshold) + (1 if produce_transmittance else 0)
        assert input_dim == 6
        assert output_dim == len(transmittance_threshold) + (1 if produce_transmittance else 0)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            PRODUCE_TRANSMITTANCE=1 if produce_transmittance else None,
        )
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.transform = transform
        self.step_size = step_size

    def clone(self,
              **kwargs) -> _core.Map:
        return TransmittanceProfiling2(
            extinction=self.extinction,
            transmittance_threshold=[self.transmittance_threshold[i] for i in range(self.output_dim)],
            step_size=self.step_size,
            transform=self.transform,
            produce_transmittance='PRODUCE_TRANSMITTANCE' in self.rdv_generics,
            **kwargs
        )


class TransmittanceProfiling3(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/transmittance_profiling_3.h',
        parameters=dict(
            extinction=_core.Map,
            step_size=float,
            transform=_torch.Tensor,
            transmittance_threshold=['OUTPUT_DIM', float]
        )
    )

    def __init__(self, extinction: _core.MapLike, transmittance_threshold: list[float], produce_transmittance: bool = False, step_size: float = 0.005, transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None:
            output_dim = len(transmittance_threshold) + (1 if produce_transmittance else 0)
        assert input_dim == 6
        assert output_dim == len(transmittance_threshold) + (1 if produce_transmittance else 0)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            PRODUCE_TRANSMITTANCE=1 if produce_transmittance else None,
        )
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.transform = transform
        self.step_size = step_size

    def clone(self,
              **kwargs) -> _core.Map:
        return TransmittanceProfiling3(
            extinction=self.extinction,
            transmittance_threshold=[self.transmittance_threshold[i] for i in range(self.output_dim)],
            step_size=self.step_size,
            transform=self.transform,
            produce_transmittance='PRODUCE_TRANSMITTANCE' in self.rdv_generics,
            **kwargs
        )


class FullProfiling(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/full_profiler_rm.h',
        parameters=dict(
            extinction=_core.Map,
            step_size=float,
            transform=_torch.Tensor,
            light_direction=_torch.Tensor,
            relative_light_direction=_torch.Tensor,
            transmittance_threshold=['OUTPUT_DIM', float]
        ),
    )

    def __init__(self, extinction: _core.MapLike, transmittance_threshold: list[float], light_direction: _torch.Tensor, relative_light_direction: _torch.Tensor, step_size: float = 0.005, transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None:
            output_dim = len(transmittance_threshold) + 5
        assert input_dim == 6
        assert output_dim == len(transmittance_threshold) + 5
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
        )
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.transform = transform
        self.light_direction = light_direction
        self.relative_light_direction = relative_light_direction
        self.step_size = step_size

    def clone(self,
              **kwargs) -> _core.Map:
        return FullProfiling(
            extinction=self.extinction,
            transmittance_threshold=[self.transmittance_threshold[i] for i in range(self.output_dim)],
            light_direction = self.light_direction,
            relative_light_direction = self.relative_light_direction,
            step_size=self.step_size,
            transform=self.transform,
            **kwargs
        )


class ProfileFT(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/profile_ft.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            transform=_torch.Tensor,
        ),
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None,
                 input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None:
            output_dim = 2
        assert input_dim == 6
        assert output_dim == 2
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
        )
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        self.majorant = majorant.cast(input_dim=6, output_dim=2)
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ProfileFT(
            extinction=self.extinction,
            majorant=self.majorant,
            transform=self.transform,
            **kwargs
        )


class ProfileFSG(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/profile_fsg.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            transform=_torch.Tensor,
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None,
                 input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        spectral_dim = scattering_albedo.output_dim
        if output_dim is not None:
            spectral_dim = output_dim - 2
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None and spectral_dim is not None:
            output_dim = spectral_dim + 2
        if output_dim is not None and spectral_dim is not None:
            assert output_dim == spectral_dim + 2, f"Output dim must be spectral dim + 2 (for extinction and anisotropy), got output_dim={output_dim} and spectral_dim={spectral_dim}"
        assert input_dim == 6
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM=spectral_dim
        )
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        self.majorant = majorant.cast(input_dim=6, output_dim=2)
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ProfileFSG(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            transform=self.transform,
            **kwargs
        )


class ProfileFDTNE(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/profile_fdtne.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            environment=_core.Map,
            environment_sampler=_core.Map,
            transform=_torch.Tensor,
            step_size=float,
            transmittance_threshold=['DEPTH_SAMPLES', float]
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 environment: _core.MapLike,
                 environment_sampler: _core.MapLike,
                 transmittance_threshold: list[float],
                 step_size: float = 0.005,
                 transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None,
                 input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        environment = _core.as_map(environment)
        environment_sampler = _core.as_map(environment_sampler)
        depth_samples = len(transmittance_threshold)
        spectral_dim = scattering_albedo.output_dim
        if spectral_dim is None:
            spectral_dim = environment.output_dim
        if spectral_dim is None and environment_sampler.output_dim is not None:
            spectral_dim = environment_sampler.output_dim - 4
        if spectral_dim is None and output_dim is not None:
            spectral_dim = (output_dim - 2 - depth_samples)//3
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None and spectral_dim is not None:
            output_dim = spectral_dim*3 + 2 + depth_samples
        if output_dim is not None and spectral_dim is not None:
            assert output_dim == spectral_dim*3 + 2 + depth_samples, f"Output dim must be spectral dim + 2 (for extinction and anisotropy), got output_dim={output_dim} and spectral_dim={spectral_dim}"
        assert input_dim == 6
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM=spectral_dim,
            DEPTH_SAMPLES=depth_samples
        )
        self.step_size = step_size
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        self.majorant = majorant.cast(input_dim=6, output_dim=2)
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        self.environment = environment.cast(input_dim=3, output_dim=spectral_dim)
        self.environment_sampler = environment_sampler.cast(input_dim=6, output_dim=None if spectral_dim is None else spectral_dim + 4)
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ProfileFDTNE(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            environment=self.environment,
            environment_sampler=self.environment_sampler,
            transmittance_threshold=self.transmittance_threshold,
            step_size=self.step_size,
            transform=self.transform,
            **kwargs
        )


class ProfileFDTNEX(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/profile_fdtnex.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            environment=_core.Map,
            environment_sampler=_core.Map,
            transform=_torch.Tensor,
            step_size=float,
            transmittance_threshold=['DEPTH_SAMPLES', float]
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 environment: _core.MapLike,
                 environment_sampler: _core.MapLike,
                 transmittance_threshold: list[float],
                 step_size: float = 0.005,
                 transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None,
                 input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        environment = _core.as_map(environment)
        environment_sampler = _core.as_map(environment_sampler)
        depth_samples = len(transmittance_threshold)
        spectral_dim = scattering_albedo.output_dim
        if spectral_dim is None:
            spectral_dim = environment.output_dim
        if spectral_dim is None and environment_sampler.output_dim is not None:
            spectral_dim = environment_sampler.output_dim - 4
        if spectral_dim is None and output_dim is not None:
            spectral_dim = (output_dim - 2 - depth_samples)//3
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None and spectral_dim is not None:
            output_dim = spectral_dim*3 + 2 + depth_samples
        if output_dim is not None and spectral_dim is not None:
            assert output_dim == spectral_dim*3 + 2 + depth_samples, f"Output dim must be spectral dim + 2 (for extinction and anisotropy), got output_dim={output_dim} and spectral_dim={spectral_dim}"
        assert input_dim == 6
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM=spectral_dim,
            DEPTH_SAMPLES=depth_samples
        )
        self.step_size = step_size
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        self.majorant = majorant.cast(input_dim=6, output_dim=2)
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        self.environment = environment.cast(input_dim=3, output_dim=spectral_dim)
        self.environment_sampler = environment_sampler.cast(input_dim=6, output_dim=None if spectral_dim is None else spectral_dim + 4)
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ProfileFDTNEX(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            environment=self.environment,
            environment_sampler=self.environment_sampler,
            transmittance_threshold=self.transmittance_threshold,
            step_size=self.step_size,
            transform=self.transform,
            **kwargs
        )


class ProfileFDTNEX2(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/profile_fdtnex2.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            environment=_core.Map,
            environment_sampler=_core.Map,
            transform=_torch.Tensor,
            step_size=float,
            transmittance_threshold=['DEPTH_SAMPLES', float]
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 environment: _core.MapLike,
                 environment_sampler: _core.MapLike,
                 transmittance_threshold: list[float],
                 step_size: float = 0.005,
                 transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None,
                 input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        environment = _core.as_map(environment)
        environment_sampler = _core.as_map(environment_sampler)
        depth_samples = len(transmittance_threshold)
        spectral_dim = scattering_albedo.output_dim
        if spectral_dim is None:
            spectral_dim = environment.output_dim
        if spectral_dim is None and environment_sampler.output_dim is not None:
            spectral_dim = environment_sampler.output_dim - 4
        if spectral_dim is None and output_dim is not None:
            spectral_dim = (output_dim - 2 - depth_samples)//3
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None and spectral_dim is not None:
            output_dim = spectral_dim*3 + 2 + depth_samples
        if output_dim is not None and spectral_dim is not None:
            assert output_dim == spectral_dim*3 + 2 + depth_samples, f"Output dim must be spectral dim + 2 (for extinction and anisotropy), got output_dim={output_dim} and spectral_dim={spectral_dim}"
        assert input_dim == 6
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM=spectral_dim,
            DEPTH_SAMPLES=depth_samples
        )
        self.step_size = step_size
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        self.majorant = majorant.cast(input_dim=6, output_dim=2)
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        self.environment = environment.cast(input_dim=3, output_dim=spectral_dim)
        self.environment_sampler = environment_sampler.cast(input_dim=6, output_dim=None if spectral_dim is None else spectral_dim + 4)
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ProfileFDTNEX2(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            environment=self.environment,
            environment_sampler=self.environment_sampler,
            transmittance_threshold=self.transmittance_threshold,
            step_size=self.step_size,
            transform=self.transform,
            **kwargs
        )



class ProfileFDTNEX3(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/profile_fdtnex3.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            environment=_core.Map,
            environment_sampler=_core.Map,
            transform=_torch.Tensor,
            step_size=float,
            environment_samples=int,
            transmittance_threshold=['DEPTH_SAMPLES', float]
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 environment: _core.MapLike,
                 environment_sampler: _core.MapLike,
                 transmittance_threshold: list[float],
                 step_size: float = 0.005,
                 environment_samples: int = 4,
                 transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None,
                 input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        environment = _core.as_map(environment)
        environment_sampler = _core.as_map(environment_sampler)
        depth_samples = len(transmittance_threshold)
        spectral_dim = scattering_albedo.output_dim
        if spectral_dim is None:
            spectral_dim = environment.output_dim
        if spectral_dim is None and environment_sampler.output_dim is not None:
            spectral_dim = environment_sampler.output_dim - 4
        if spectral_dim is None and output_dim is not None:
            spectral_dim = (output_dim - 2 - depth_samples)//3
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None and spectral_dim is not None:
            output_dim = spectral_dim*3 + 2 + depth_samples
        if output_dim is not None and spectral_dim is not None:
            assert output_dim == spectral_dim*3 + 2 + depth_samples, f"Output dim must be spectral dim + 2 (for extinction and anisotropy), got output_dim={output_dim} and spectral_dim={spectral_dim}"
        assert input_dim == 6
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM=spectral_dim,
            DEPTH_SAMPLES=depth_samples
        )
        self.step_size = step_size
        self.environment_samples = environment_samples
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        self.majorant = majorant.cast(input_dim=6, output_dim=2)
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        self.environment = environment.cast(input_dim=3, output_dim=spectral_dim)
        self.environment_sampler = environment_sampler.cast(input_dim=6, output_dim=None if spectral_dim is None else spectral_dim + 4)
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ProfileFDTNEX3(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            environment=self.environment,
            environment_sampler=self.environment_sampler,
            transmittance_threshold=self.transmittance_threshold,
            step_size=self.step_size,
            environment_samples=self.environment_samples,
            transform=self.transform,
            **kwargs
        )


class ProfileFDTNEX4(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/profile_fdtnex4.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            environment=_core.Map,
            environment_sampler=_core.Map,
            transform=_torch.Tensor,
            step_size=float,
            environment_samples=int,
            transmittance_threshold=['DEPTH_SAMPLES', float]
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 environment: _core.MapLike,
                 environment_sampler: _core.MapLike,
                 transmittance_threshold: list[float],
                 step_size: float = 0.005,
                 environment_samples: int = 4,
                 transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None,
                 input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        environment = _core.as_map(environment)
        environment_sampler = _core.as_map(environment_sampler)
        depth_samples = len(transmittance_threshold)
        spectral_dim = scattering_albedo.output_dim
        if spectral_dim is None:
            spectral_dim = environment.output_dim
        if spectral_dim is None and environment_sampler.output_dim is not None:
            spectral_dim = environment_sampler.output_dim - 4
        if spectral_dim is None and output_dim is not None:
            spectral_dim = (output_dim - 2 - depth_samples)//3
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None and spectral_dim is not None:
            output_dim = spectral_dim*3 + 2 + depth_samples
        if output_dim is not None and spectral_dim is not None:
            assert output_dim == spectral_dim*3 + 2 + depth_samples, f"Output dim must be spectral dim + 2 (for extinction and anisotropy), got output_dim={output_dim} and spectral_dim={spectral_dim}"
        assert input_dim == 6
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM=spectral_dim,
            DEPTH_SAMPLES=depth_samples
        )
        self.step_size = step_size
        self.environment_samples = environment_samples
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        self.majorant = majorant.cast(input_dim=6, output_dim=2)
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        self.environment = environment.cast(input_dim=3, output_dim=spectral_dim)
        self.environment_sampler = environment_sampler.cast(input_dim=6, output_dim=None if spectral_dim is None else spectral_dim + 4)
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ProfileFDTNEX4(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            environment=self.environment,
            environment_sampler=self.environment_sampler,
            transmittance_threshold=self.transmittance_threshold,
            step_size=self.step_size,
            environment_samples=self.environment_samples,
            transform=self.transform,
            **kwargs
        )


class ProfileFDTNEX5(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/profile_fdtnex5.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            environment=_core.Map,
            environment_sampler=_core.Map,
            transform=_torch.Tensor,
            step_size=float,
            environment_samples=int,
            transmittance_threshold=['DEPTH_SAMPLES', float]
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 environment: _core.MapLike,
                 environment_sampler: _core.MapLike,
                 transmittance_threshold: list[float],
                 step_size: float = 0.005,
                 environment_samples: int = 4,
                 transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None,
                 input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        environment = _core.as_map(environment)
        environment_sampler = _core.as_map(environment_sampler)
        depth_samples = len(transmittance_threshold)
        spectral_dim = scattering_albedo.output_dim
        if spectral_dim is None:
            spectral_dim = environment.output_dim
        if spectral_dim is None and environment_sampler.output_dim is not None:
            spectral_dim = environment_sampler.output_dim - 4
        if spectral_dim is None and output_dim is not None:
            spectral_dim = (output_dim - 2 - depth_samples)//3
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None and spectral_dim is not None:
            output_dim = spectral_dim*3 + 2 + depth_samples
        if output_dim is not None and spectral_dim is not None:
            assert output_dim == spectral_dim*3 + 2 + depth_samples, f"Output dim must be spectral dim + 2 (for extinction and anisotropy), got output_dim={output_dim} and spectral_dim={spectral_dim}"
        assert input_dim == 6
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM=spectral_dim,
            DEPTH_SAMPLES=depth_samples
        )
        self.step_size = step_size
        self.environment_samples = environment_samples
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        self.majorant = majorant.cast(input_dim=6, output_dim=2)
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        self.environment = environment.cast(input_dim=3, output_dim=spectral_dim)
        self.environment_sampler = environment_sampler.cast(input_dim=6, output_dim=None if spectral_dim is None else spectral_dim + 4)
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ProfileFDTNEX5(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            environment=self.environment,
            environment_sampler=self.environment_sampler,
            transmittance_threshold=self.transmittance_threshold,
            step_size=self.step_size,
            environment_samples=self.environment_samples,
            transform=self.transform,
            **kwargs
        )


class ProfileFDTNEX6(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/profile_fdtnex6.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            environment=_core.Map,
            environment_sampler=_core.Map,
            transform=_torch.Tensor,
            step_size=float,
            environment_samples=int,
            transmittance_threshold=['DEPTH_SAMPLES', float]
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 environment: _core.MapLike,
                 environment_sampler: _core.MapLike,
                 transmittance_threshold: list[float],
                 step_size: float = 0.005,
                 environment_samples: int = 4,
                 transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None,
                 input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        environment = _core.as_map(environment)
        environment_sampler = _core.as_map(environment_sampler)
        depth_samples = len(transmittance_threshold)
        spectral_dim = scattering_albedo.output_dim
        if spectral_dim is None:
            spectral_dim = environment.output_dim
        if spectral_dim is None and environment_sampler.output_dim is not None:
            spectral_dim = environment_sampler.output_dim - 4
        if spectral_dim is None and output_dim is not None:
            spectral_dim = (output_dim - 2 - depth_samples)//3
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None and spectral_dim is not None:
            output_dim = spectral_dim*3 + 2 + depth_samples
        if output_dim is not None and spectral_dim is not None:
            assert output_dim == spectral_dim*3 + 2 + depth_samples, f"Output dim must be spectral dim + 2 (for extinction and anisotropy), got output_dim={output_dim} and spectral_dim={spectral_dim}"
        assert input_dim == 6
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM=spectral_dim,
            DEPTH_SAMPLES=depth_samples
        )
        self.step_size = step_size
        self.environment_samples = environment_samples
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        self.majorant = majorant.cast(input_dim=6, output_dim=2)
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        self.environment = environment.cast(input_dim=3, output_dim=spectral_dim)
        self.environment_sampler = environment_sampler.cast(input_dim=6, output_dim=None if spectral_dim is None else spectral_dim + 4)
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ProfileFDTNEX6(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            environment=self.environment,
            environment_sampler=self.environment_sampler,
            transmittance_threshold=self.transmittance_threshold,
            step_size=self.step_size,
            environment_samples=self.environment_samples,
            transform=self.transform,
            **kwargs
        )


class ProfileFDTNEX7(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/profile_fdtnex7.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            environment=_core.Map,
            environment_sampler=_core.Map,
            transform=_torch.Tensor,
            step_size=float,
            environment_samples=int,
            transmittance_threshold=['DEPTH_SAMPLES', float]
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 environment: _core.MapLike,
                 environment_sampler: _core.MapLike,
                 transmittance_threshold: list[float],
                 step_size: float = 0.005,
                 environment_samples: int = 4,
                 transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None,
                 input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        environment = _core.as_map(environment)
        environment_sampler = _core.as_map(environment_sampler)
        depth_samples = len(transmittance_threshold)
        spectral_dim = scattering_albedo.output_dim
        if spectral_dim is None:
            spectral_dim = environment.output_dim
        if spectral_dim is None and environment_sampler.output_dim is not None:
            spectral_dim = environment_sampler.output_dim - 4
        if spectral_dim is None and output_dim is not None:
            spectral_dim = (output_dim - 2 - depth_samples)//3
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None and spectral_dim is not None:
            output_dim = spectral_dim*3 + 2 + depth_samples
        if output_dim is not None and spectral_dim is not None:
            assert output_dim == spectral_dim*3 + 2 + depth_samples, f"Output dim must be spectral dim + 2 (for extinction and anisotropy), got output_dim={output_dim} and spectral_dim={spectral_dim}"
        assert input_dim == 6
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM=spectral_dim,
            DEPTH_SAMPLES=depth_samples
        )
        self.step_size = step_size
        self.environment_samples = environment_samples
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        self.majorant = majorant.cast(input_dim=6, output_dim=2)
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        self.environment = environment.cast(input_dim=3, output_dim=spectral_dim)
        self.environment_sampler = environment_sampler.cast(input_dim=6, output_dim=None if spectral_dim is None else spectral_dim + 4)
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ProfileFDTNEX7(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            environment=self.environment,
            environment_sampler=self.environment_sampler,
            transmittance_threshold=self.transmittance_threshold,
            step_size=self.step_size,
            environment_samples=self.environment_samples,
            transform=self.transform,
            **kwargs
        )


class ProfileFDTNEX8(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/volume_rendering/profile_fdtnex8.h',
        parameters=dict(
            extinction=_core.Map,
            majorant=_core.Map,
            scattering_albedo=_core.Map,
            anisotropy=_core.Map,
            environment=_core.Map,
            environment_sampler=_core.Map,
            transform=_torch.Tensor,
            step_size=float,
            environment_samples=int,
            sample_index=int,
            transmittance_threshold=['DEPTH_SAMPLES', float]
        ),
        stochastic=True,
    )

    def __init__(self,
                 extinction: _core.MapLike,
                 majorant: _core.MapLike,
                 scattering_albedo: _core.MapLike,
                 anisotropy: _core.MapLike,
                 environment: _core.MapLike,
                 environment_sampler: _core.MapLike,
                 transmittance_threshold: list[float],
                 step_size: float = 0.005,
                 environment_samples: int = 4,
                 sample_index: int = 0,
                 transform: _core.TensorLike | _core.deferred | None = None, input_dim=None, output_dim=None,
                 input_requires_grad=False, bw_uses_output=False):
        extinction = _core.as_map(extinction)
        majorant = _core.as_map(majorant)
        scattering_albedo = _core.as_map(scattering_albedo)
        anisotropy = _core.as_map(anisotropy)
        environment = _core.as_map(environment)
        environment_sampler = _core.as_map(environment_sampler)
        depth_samples = len(transmittance_threshold)
        spectral_dim = scattering_albedo.output_dim
        if spectral_dim is None:
            spectral_dim = environment.output_dim
        if spectral_dim is None and environment_sampler.output_dim is not None:
            spectral_dim = environment_sampler.output_dim - 4
        if spectral_dim is None and output_dim is not None:
            spectral_dim = (output_dim - 4 - depth_samples)//3
        if transform is None:
            transform = _vk.mat4x3.trs()
        if input_dim is None:
            input_dim = 6
        if output_dim is None and spectral_dim is not None:
            output_dim = spectral_dim*3 + 4 + depth_samples
        if output_dim is not None and spectral_dim is not None:
            assert output_dim == spectral_dim*3 + 4 + depth_samples, f"Output dim must be spectral dim + 2 (for extinction and anisotropy), got output_dim={output_dim} and spectral_dim={spectral_dim}"
        assert input_dim == 6
        scattering_albedo = scattering_albedo.cast(input_dim=3, output_dim=spectral_dim)
        super().__init__(
            input_dim=input_dim, output_dim=output_dim,
            input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output,
            SPECTRAL_DIM=spectral_dim,
            DEPTH_SAMPLES=depth_samples
        )
        self.step_size = step_size
        self.environment_samples = environment_samples
        self.sample_index = sample_index
        for i, t in enumerate(transmittance_threshold):
            self.transmittance_threshold[i] = t
        self.extinction = extinction.cast(input_dim=3, output_dim=1)
        self.majorant = majorant.cast(input_dim=6, output_dim=2)
        self.scattering_albedo = scattering_albedo
        self.anisotropy = anisotropy.cast(input_dim=3, output_dim=1)
        self.environment = environment.cast(input_dim=3, output_dim=spectral_dim)
        self.environment_sampler = environment_sampler.cast(input_dim=6, output_dim=None if spectral_dim is None else spectral_dim + 4)
        self.transform = transform

    def clone(self,
              **kwargs) -> _core.Map:
        return ProfileFDTNEX8(
            extinction=self.extinction,
            majorant=self.majorant,
            scattering_albedo=self.scattering_albedo,
            anisotropy=self.anisotropy,
            environment=self.environment,
            environment_sampler=self.environment_sampler,
            transmittance_threshold=self.transmittance_threshold,
            step_size=self.step_size,
            environment_samples=self.environment_samples,
            sample_index=self.sample_index,
            transform=self.transform,
            **kwargs
        )




class _MajorantMergeCompute(_core.Compute):
    __extension_info__ = dict(
        parameters=dict(
            in_majorant=_torch.Tensor,
            in_radius=_torch.Tensor,
            out_majorant=_torch.Tensor,
            out_radius=_torch.Tensor,
            shape=[3, int],
            radius=int,
            cell_size = float,
        ),
    code="""
int get_index(int x, int y, int z) {
    x = max(0, min(x, parameters.shape[2] - 1));
    y = max(0, min(y, parameters.shape[1] - 1));
    z = max(0, min(z, parameters.shape[0] - 1));
    return x + parameters.shape[2] * (y + parameters.shape[1] * z);
}

MAIN(tid) {
    // PRINT("Hola %d", parameters.radius);
    float_ptr in_majorant = float_ptr(parameters.in_majorant);
    int_ptr in_radius = int_ptr(parameters.in_radius);
    float_ptr out_majorant = float_ptr(parameters.out_majorant);
    int_ptr out_radius = int_ptr(parameters.out_radius);
    // keep the original value by default
    out_majorant.data[tid.x] = in_majorant.data[tid.x];
    out_radius.data[tid.x] = in_radius.data[tid.x];
    float majorant = 0.0; //in_majorant.data[tid.x];
    int x = int(tid.x) % parameters.shape[2];
    int y = int(tid.x / parameters.shape[2]) % parameters.shape[1];
    int z = int(tid.x) / (parameters.shape[2] * parameters.shape[1]);
    bool can_expand = true;
    for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                can_expand = can_expand && in_radius.data[get_index(x + dx*parameters.radius, y + dy*parameters.radius, z + dz*parameters.radius)] == parameters.radius;
                majorant = max(majorant, in_majorant.data[get_index(x + dx*parameters.radius, y + dy*parameters.radius, z + dz*parameters.radius)]);
            }
    if (!can_expand)
    return;
    for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                float dif = majorant - in_majorant.data[get_index(x + dx*parameters.radius, y + dy*parameters.radius, z + dz*parameters.radius)];
                can_expand = can_expand && dif <= 1 / (parameters.cell_size * parameters.radius);
            }
    if (!can_expand)
    return;
    out_majorant.data[tid.x] = majorant;
    out_radius.data[tid.x] = parameters.radius * 2;    
}
        """
    )

    def bind(self, *args, **kwargs) -> _core.ComputeTask:
        in_majorant, in_radius, current_radius = args
        task = _MajorantMergeCompute.create_task(in_majorant.numel())
        task.binder.in_majorant = _vk.wrap_gpu(in_majorant, mode='in')
        task.binder.in_radius = _vk.wrap_gpu(in_radius, mode='in')
        task.binder.out_majorant = _vk.wrap_gpu(_vk.zeros_like(in_majorant), mode='out')
        task.binder.out_radius = _vk.wrap_gpu(_vk.zeros_like(in_radius), mode='out')
        for i, d in enumerate(in_majorant.shape[:3]):
            task.binder.shape[i] = d
        task.binder.radius = current_radius
        task.binder.cell_size = 2.0 / max(in_majorant.shape)
        return task

    def result(self, task: _core.ComputeTask):
        return task.binder.out_majorant.unwrap(), task.binder.out_radius.unwrap()


def _majorant_merge(in_majorant, in_radius, current_radius):
    return _MajorantMergeCompute.eval(in_majorant, in_radius, current_radius)


def majorant_grid(grid: _torch.Tensor):
    """
    Given a grid, it is assumed to be mapping from -1 to 1.
    A majorant grid with majorant and radius per sample is computed conservatively.
    """
    # Extract the maximum value in each 2x2x2 block, and assign it to the center of the block
    import torch.nn.functional as F
    max_grid = F.max_pool3d(
        grid.permute(3, 0, 1, 2).unsqueeze(0),
        kernel_size=2,
        stride=2,
        padding = 1
    )
    # initial Majorant grid and radius grid
    majorant_grid = max_grid[0].permute(1, 2, 3, 0).to(_core.device())
    radius_grid = _torch.ones_like(majorant_grid, dtype=_torch.int32)
    # Merge majorants the radii of the blocks
    max_dim = max(majorant_grid.shape[:3])
    r = 1
    while r < max_dim:
        majorant_grid, radius_grid = _majorant_merge(majorant_grid, radius_grid, current_radius=r)
        r *= 2
    cell_size = .5 / max_dim
    majorant_grid = _torch.cat([majorant_grid, radius_grid.float() * cell_size], dim=-1)
    majorant_grid = _vk.tensor_copy(majorant_grid)
    return majorant_grid


