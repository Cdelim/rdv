import numpy as _np
import torch as _torch
import os as _os
import threading as _threading

import vulky as _vk
import typing as _typing
import enum as _enum
from functools import cached_property
import weakref as _weakref


__SEEDS_POOL__ = None
__INCLUDE_PATH__ = _os.path.dirname(__file__).replace('\\','/') + '/include'
__TORCH_DEVICE__ = _torch.device('cuda:0') if _torch.cuda.is_available() else _torch.device('cpu')  #TODO: Check with AMD
__RDV_PATH__ = _os.path.dirname(__file__)

__RDV_PROFILER__ = None


MapLike = _typing.Union[None, 'Map', int, float, _torch.Tensor, _typing.List[_typing.Any], _typing.Tuple[_typing.Any, ...]]
"""
Helper type for values that can be converted to a Map.
"""

TensorLike = _typing.Union[int, float, _torch.Tensor, _typing.List[_typing.Any], _typing.Tuple[_typing.Any, ...]]
"""
Helper type for values that can be converted to a Tensor.
"""



def set_profiler(prof):
    global __RDV_PROFILER__
    __RDV_PROFILER__ = prof


def device() -> _torch.device:
    """
    Gets the torch device visible by vulkan backend.
    Ensure tensors are valid for rdv compute and maps using:
    >>> t.to(rdv.device())
    """
    return __TORCH_DEVICE__


def manual_seed(seed: _typing.Optional[int] = None):
    """
    Sets the seed used for the generation of torch and rdv randoms.
    Useful for replication.

    Example
    -------
    >>> import torch
    >>> import rdv
    >>> rdv.manual_seed(5)
    >>> print(rdv.randn.cast(output_dim=3)(torch.zeros(10,1)))
    """
    _torch.manual_seed(seed)
    global __SEEDS_POOL__
    __SEEDS_POOL__ = None

def _generate_seeds() -> _typing.Tuple[int, int, int, int]:
    """
    Creates the new 4 int seed for hybrid-Taus rgn algorithm.
    """
    global __SEEDS_POOL__
    if __SEEDS_POOL__ is None or len(__SEEDS_POOL__) == 0:
        __SEEDS_POOL__ = list(_torch.randint(129, 1 << 30, size=(1024*32, 4)))
    r = __SEEDS_POOL__.pop()
    return r[0].item(), r[1].item(), r[2].item(), r[3].item()


def time_check(tag: str = ""):
    import time
    class MeassurementContext:
        def __enter__(self):
            self.start_time = time.perf_counter()
        def __exit__(self, exc_type, exc_val, exc_tb):
            print(f"{tag}:{time.perf_counter() - self.start_time}s")
    return MeassurementContext()

# Read compute template
with open(__RDV_PATH__ + '/include/compute_template.h') as f:
    __COMPUTE_TEMPLATE__ = f.read()


class BACKWARD_IMPLEMENTATIONS(_enum.IntEnum):
    NONE = 0
    """
    The map source code doesn't contains any backward function.
    """
    DEFAULT = 1
    """
    The map source code contains only the default backward function (input, output_grad, input_grad).
    """
    WITH_OUTPUT = 2
    """
    The map source code contains only the output provided backward function (input, output, output_grad, input_grad).
    """
    ALL = 3
    """
    The map source code contains both backward functions, with and without output.
    """


class _DeferredParametersManager:

    rdv_named_tensors_buffer = None
    rdv_named_tensors_info = { }  # map from key to { key_id:int, references:int } representing id and references count
    rdv_named_tensors_free_ids = []  # reusable ids for named tensors
    rdv_wrapped_tensors = {}  # wrapped tensors bound
    rdv_wrapped_grads = {}  # wrapped grads created

    @classmethod
    def init(cls):
        cls.rdv_named_tensors_buffer = _vk.object_buffer(layout=_vk.Layout.from_description(_vk.LayoutAlignment.SCALAR,
        dict(
            data=[1024, _torch.int64],  # data ptr of the contiguous bound tensor
            grad_data=[1024, _torch.int64],
        )))
        cls.rdv_is_initialized = True

    @classmethod
    def resolve(cls, key: str, offset: int) -> int:
        if key not in cls.rdv_named_tensors_info:
            if len(cls.rdv_named_tensors_free_ids) > 0:
                key_id = cls.rdv_named_tensors_free_ids.pop()
            else:
                key_id = len(cls.rdv_named_tensors_info)
            cls.rdv_named_tensors_info[key] = { 'key_id':key_id, 'references': 1}
        else:
            key_id = cls.rdv_named_tensors_info[key]['key_id']
            cls.rdv_named_tensors_info[key]['references'] += 1
        return key_id

    @classmethod
    def free(cls, key: str, offset: int):
        assert key in cls.rdv_named_tensors_info
        info = cls.rdv_named_tensors_info[key]
        info['references'] -= 1
        if info['references'] == 0:  # key info index can be reused
            cls.rdv_named_tensors_info.pop(key)
            cls.rdv_named_tensors_free_ids.append(info['key_id'])

    @classmethod
    def bind(cls, named_tensors: _typing.Dict[str, _torch.Tensor], compute_grads: bool = False) -> _typing.Dict[str, _torch.Tensor]:
        grads = { }
        with cls.rdv_named_tensors_buffer as b:
            grad_data = b.grad_data
            data = b.data
            for key, t in named_tensors.items():
                key_id = cls.rdv_named_tensors_info[key]['key_id']
                wgpu = _vk.wrap_gpu(t, 'in')
                cls.rdv_wrapped_tensors[key_id] = wgpu
                data[key_id] = wgpu
                if t is not None:
                    if compute_grads:
                        if t.requires_grad:
                            g = _vk.zeros_like(t)
                            wgpu = _vk.wrap_gpu(g, 'inout')
                            cls.rdv_wrapped_grads[key_id] = wgpu
                            grad_data[key_id] = wgpu  # valid for a vulkan tensor
                            grads[key] = g
        return grads

    @classmethod
    def unbind(cls, named_tensors: _typing.Dict[str, _torch.Tensor], grads: bool = False):
        for key, t in named_tensors.items():
            key_id = cls.rdv_named_tensors_info[key]['key_id']
            cls.rdv_wrapped_tensors[key_id].unwrap()
            if grads and key_id in cls.rdv_wrapped_grads:
                cls.rdv_wrapped_grads[key_id].unwrap()


    @classmethod
    def validate(cls, named_tensors: _typing.Dict[str, _torch.Tensor], *deferred):
        pass  # TODO: Implement this to serve as a debug tool


class deferred:
    def __init__(self, key: str, shape: _typing.Tuple[int,...], index: _typing.Optional[int] = None):
        self._key = key
        if any(s <= 0 for s in shape):
            assert index is None, f"Invalid shape {shape} for deferred parameter. Indexed deferred parameters requires all dimensions in shape to be positive."
        import math
        self._offset = 0 if index is None else math.prod(shape) * index
        self._index = index
        self._shape = shape
        self.id = _DeferredParametersManager.resolve(key, self._offset)

    @cached_property
    def as_uint64(self):
        return (self.id << 1 | 1) | (self._offset << 32)

    @cached_property
    def shape(self):
        return self._shape

    # def cast(self, map_dim: int) -> 'deferred':
    #     assert isinstance(map_dim, int) and map_dim > 0, "map_dim must be a valid int"
    #     assert self._map_dim is None or self._map_dim == map_dim, f"Can not change an existing map dim {self._map_dim} for {map_dim}"
    #     if self._map_dim == map_dim:
    #         return self
    #     return deferred(self._key, map_dim, self._indices)
    #
    # @cached_property
    # def is_fixed(self) -> bool:
    #     return self._map_dim is not None

    @cached_property
    def dimension(self) -> int:
        return len(self._shape)

    def __del__(self):
        _DeferredParametersManager.free(self._key, self._offset)

    def evaluate(self, **deferred_parameters: _torch.Tensor) -> _torch.Tensor:
        assert self._key in deferred_parameters, f"Deferred parameter with key {self._key} not found in provided parameters."
        base_tensor = deferred_parameters[self._key]
        if self._index is None:
            return base_tensor
        return base_tensor.view(-1, *self._shape)[self._index]



def as_tensor(t: TensorLike) -> _torch.Tensor:
    if not isinstance(t, _torch.Tensor):
        if isinstance(t, float) or isinstance(t, int):
            t = [t]
        t = _vk.tensor_copy(_torch.as_tensor(t, device=device()))
    return t


def as_tensor_or_deferred(t: TensorLike | deferred) -> _torch.Tensor | deferred:
    if isinstance(t, deferred):
        return t
    return as_tensor(t)

def ensure_tensor(t: _typing.Union[TensorLike, deferred], map_dim: int):
    t = as_tensor_or_deferred(t)
    assert len(t.shape) == map_dim, f"Tensor or deferred with shape {t.shape} can not be used in a map requiring dimension {map_dim}"
    if isinstance(t, deferred):
        return t
    assert not t.requires_grad, "Tensors bound directly can not require gradients. Use deferred parameters instead."
    while len(t.shape) < map_dim:
        t = t.unsqueeze(0)
    while len(t.shape) > map_dim:
        try:
            t = t.squeeze(0)
        except:
            raise ValueError(f"Tensor with shape {t.shape} can not be reduced to map dimension {map_dim}")
    return t


MeshInfo = dict(
    __name__='MeshInfo',
    positions=_torch.Tensor,
    normals=_torch.Tensor,
    coordinates=_torch.Tensor,
    tangents=_torch.Tensor,
    binormals=_torch.Tensor,
    indices=_torch.Tensor
)
"""
Represents a mesh object in a compute.
"""


RaycastableInfo = dict(
    __name__='RaycastableInfo',
    callable_map=_torch.int64,
    explicit_info=_torch.int64,
)
"""
Raycastable objects in rdv are a union between meshes or AABBs (explicit info) and callable maps acting as geometries.
"""


class SampleLocation(_enum.IntEnum):
    """
    Enumeration for sample locations in sampling maps.
    Used during sensor captures to define where samples are taken from in cells.
    """
    CENTER = 0,
    """
    The sample is taken from the center of the cell.
    """
    CORNER = 1,
    """
    Samples are taken from the corners of the cell.
    """
    RANDOM = 2,
    """
    Samples are taken from random positions inside the cell.
    """


class KernelInfo:
    input_dim: int
    output_dim: int
    codename: str
    input_requires_grad: bool
    bw_uses_output: bool
    stochastic: bool
    code: str
    include_dirs: _typing.Set[str]


class Singleton:
    """
    Simple singleton pattern base class for singleton maps.
    """
    __instance__ = None
    @classmethod
    def get_instance(cls):
        if cls.__instance__ is None:
            cls.__instance__ = cls()
        return cls.__instance__



class _DispatcherEngine(object):
    __COMPUTE_IDS__ = 0  # auto-increment id for all different kernel codes generated in the app.
    __COMPUTE_ID_BY_SIGNATURE__ = {}  # Compute id for a map signature. From tuple with signature to id.
    __KERNELS__ :_typing.Dict[int, KernelInfo] = {}  # Kernel info for each kernel id.
    __DEFINED_STRUCTS__ = {}  # All defined structures
    __BUILTIN_STRUCTS__ = []  # Defined structs in core.h
    __ENGINE_OBJECTS__ = None  # Objects to dispatch map evaluation and raycasting
    __EVAL_PIPELINES__ = {}  # From static signature to pipeline object
    __EVAL_MANAGERS__ = {}  # From dispatch signature to pipeline object
    __SYSTEM_BUFFER__ = None  # Object buffer with system values on it.
    __RANDOMS__ = []  # Pre-generated randoms for system buffer
    __IMAGES__ = [[], [], []]  # Registered images for evaluation, separated by dimension (1D, 2D, 3D)
    __IMAGES_REUSABLE_INDICES__ = [[], [], []]  # Reusable indices for registered images, separated by dimension (1D, 2D, 3D)

    @classmethod
    def generate_map_kernels(cls, compute_ids: _typing.Iterable[int]) -> (str, _typing.Set[str]):
        """
        Generates the code for evaluating all maps provided and their dependencies.
        Returns the code and the list of required dirs to include.
        """
        codes = []
        for k, s in cls.__DEFINED_STRUCTS__.items():
            if k not in cls.__BUILTIN_STRUCTS__:
                codes.append(s)  # append all external defined structs not matter dependences
        include_dirs = set()
        for m in sorted(compute_ids):
            ki = cls.__KERNELS__[m]
            include_dirs.update(ki.include_dirs)
            codes.append(ki.code)
        return '\n\r'.join(codes), include_dirs

    @classmethod
    def create_code_for_dynamic_calls(cls, map: 'Map', input_dim, output_dim, input_requires, bw_uses_output):
        fw_cases = ""
        bw_cases = ""
        children_kernel_ids = set(d.rdv_kernel_id for d in map.children)
        input_grad_parameter = f", inout float _input_grad[{input_dim}]" if input_requires else ""
        output_parameter = f", in float _output[{output_dim}]" if bw_uses_output else ""
        for id in children_kernel_ids:
            ki = cls.__KERNELS__[id]
            code_name = ki.codename
            if ki.input_dim == input_dim and ki.output_dim == output_dim and ki.input_requires_grad == input_requires and ki.bw_uses_output == bw_uses_output:
                fw_cases += f"""
                    case {id}: forward({code_name}(buffer_{code_name}(dynamic_map)), _input, _output); break;
                    """
                bw_cases += f"""
                    case {id}: backward({code_name}(buffer_{code_name}(dynamic_map)), _input{output_parameter}, _output_grad{input_grad_parameter}); break;
                    """
        return f"""
    void dynamic_forward (MAP_DECL, GPUPtr dynamic_map, in float _input[{input_dim}], out float _output[{output_dim}]) {{
        for (int i=0; i<{output_dim}; i++) _output[i] = 0.0;//((i^13+15 + int(random()*17))%{output_dim})/float({output_dim});
        if (dynamic_map == 0) {{
            return;
        }}
        int map_id = int_ptr(dynamic_map).data[0];
        switch(map_id)
        {{
        {fw_cases}
        }}  
    }}

    void dynamic_backward(MAP_DECL, GPUPtr dynamic_map, in float _input[{input_dim}]{output_parameter}, in float _output_grad[{output_dim}]{input_grad_parameter})  {{
        if (dynamic_map == 0) return;
        int map_id = int_ptr(dynamic_map).data[0];
        switch(map_id)
        {{
        {bw_cases}
        }}  
    }}
            """

    @classmethod
    def generate_single_map_kernel(cls, map: 'Map', kernel_id: int):
        """
        Generates the kernel of a specific map.
        """
        map_kernel_info = cls.__KERNELS__[kernel_id]
        codename = map_kernel_info.codename
        code = ""
        map_object_parameters_code, external_structs, _ = cls.create_code_type_definition(map.rdv_type_definition,
                                                                                          map.rdv_generics,
                                                                                          map.rdv_parameters,
                                                                                          is_first_block=True)
        for struct_name, struct_code in external_structs.items():
            if struct_name in cls.__DEFINED_STRUCTS__:
                assert cls.__DEFINED_STRUCTS__[
                           struct_name] == struct_code, f'A different body was already defined for {struct_name}'
            else:
                # code += struct_code + "\n"  # only save in defined structs
                cls.__DEFINED_STRUCTS__[struct_name] = struct_code
        # Add buffer_reference definition with codename and map object layout
        code += f"#define RDV_CODENAME {codename}"
        code += f"""
    layout(buffer_reference, scalar, buffer_reference_align=8) buffer MAP_BUFFER_NAME(RDV_CODENAME) {{{map_object_parameters_code}}};
    struct RDV_CODENAME {{ MAP_BUFFER_NAME(RDV_CODENAME) data; }};
    """
        for g, v in map.rdv_generics.items():
            code += f"#define {g} {v} \n"
        code += f"#define MAP_DECL in RDV_CODENAME _this \n"
        code += f"#define parameters _this.data \n"
        code += f"#include \"system\\push_signatures.h\"\n"

        for s in map.rdv_dynamic_requires:  # Generate dynamic access code for all required signatures
            code += cls.create_code_for_dynamic_calls(map, *s)

        code += map.rdv_source_code + "\n"

        code += f"#include \"system\\pop_signatures.h\"\n"

        code += f"#undef MAP_DECL\n"
        code += f"#undef RDV_CODENAME\n"
        code += f"#undef parameters\n"
        for g in map.rdv_generics:
            code += f"#undef {g}\n"
        map_kernel_info.code = code

    @classmethod
    def generate_compute_kernel(cls, compute: 'Compute', task: 'ComputeTask'):
        """
        Generates the kernel of a specific compute and task.
        """
        code = ""
        compute_parameters_code, external_structs, _ = cls.create_code_type_definition(
            compute.rdv_type_definition,
            {**compute.rdv_generics, **task.rdv_generics},
            task.binder,
            is_first_block=True)
        for struct_name, struct_code in external_structs.items():
            if struct_name in cls.__DEFINED_STRUCTS__:
                assert cls.__DEFINED_STRUCTS__[
                           struct_name] == struct_code, f'A different body was already defined for {struct_name}'
            else:
                # code += struct_code + "\n"  # only save in defined structs
                cls.__DEFINED_STRUCTS__[struct_name] = struct_code
        # Add buffer_reference definition with codename and map object layout
        code += f"""
            layout(binding=2, scalar, buffer_reference_align=8) uniform ComputeParameters {{{compute_parameters_code}}} parameters;
            """
        for g, v in task.rdv_generics.items():
            code += f"#define {g} {v} \n"

        code += compute.rdv_source_code + "\n"
        for g in compute.rdv_generics:
            code += f"#undef {g}\n"
        return code

    @classmethod
    def register_instance(cls, map: 'Map'):  # sets a new or existing compute id for the map
        """
        Registers a map if new signature.
        Returns the id of the map and the code name
        """
        signature = map.rdv_signature
        compute_id = cls.__COMPUTE_ID_BY_SIGNATURE__.get(signature)
        if compute_id is None:
            cls.__COMPUTE_IDS__ += 1
            compute_id = cls.__COMPUTE_IDS__
            cls.__COMPUTE_ID_BY_SIGNATURE__[signature] = compute_id
            map_kernel_info = KernelInfo()
            cls.__KERNELS__[compute_id] = map_kernel_info
            map_kernel_info.codename = f"{(type(map).__name__).replace('_', '')}_{compute_id}"  # 'rdv_map_' + str(instance_id)
            map_kernel_info.input_dim = map.input_dim
            map_kernel_info.output_dim = map.output_dim
            map_kernel_info.include_dirs = map.rdv_include_dirs
            map_kernel_info.input_requires_grad = map.input_requires_grad
            map_kernel_info.bw_uses_output = map.bw_uses_output
            map_kernel_info.stochastic = map.is_stochastic
            cls.generate_single_map_kernel(map, compute_id)  # generate code for the map kernel
        return compute_id

    @classmethod
    def create_code_type_definition(cls, type_definition, generics, field_value = None, is_first_block = False):
        if type_definition == Map:
            assert field_value is not None, "Basic structs can not bind explicit maps. Use int64_t if you want a reference."
            return cls.__KERNELS__[field_value.rdv_kernel_id].codename, {}, []
        if type_definition == _torch.Tensor:
            return 'GPUPtr', {}, []
        if _vk.Layout.is_scalar_type(type_definition):
            if type_definition == int:
                return 'int', {}, []
            if type_definition == float:
                return 'float', {}, []
            return {
                _torch.int32: 'int',
                _torch.float32: 'float',
                _torch.int64: 'GPUPtr'
            }[type_definition], {}, []
        if isinstance(type_definition, list):
            size = type_definition[0] if isinstance(type_definition[0], int) else generics[type_definition[0]]
            t = type_definition[1]
            t_value = None if field_value is None else field_value[0]
            element_decl, inner_structures, element_sizes = cls.create_code_type_definition(t, generics, t_value)
            return element_decl, inner_structures, [size] + element_sizes
        if isinstance(type_definition, dict):
            inner_structures = {}
            if '__name__' in type_definition.keys():  # external struct
                struct_code = f"struct {type_definition['__name__']} {{"
                for field_id, field_type in type_definition.items():
                    if field_id != '__name__':
                        t, field_inner_structures, sizes = cls.create_code_type_definition(field_type, {})
                        struct_code += t + " " + field_id + ''.join(f"[{size}]" for size in sizes) + '; \n'
                        inner_structures.update(field_inner_structures)
                struct_code += '};'
                inner_structures[type_definition['__name__']] = struct_code
                return type_definition['__name__'], inner_structures, []
            else:  # block
                assert is_first_block, 'Can not create a nested block. Add a name attribute to the dictionary to make it a struct'
                code = ""
                for field_id, field_type in type_definition.items():
                    f_value = None if field_value is None else getattr(field_value, field_id)
                    t, field_inner_structures, sizes = cls.create_code_type_definition(field_type, generics, f_value)
                    code += t + " " + field_id + ''.join(f"[{size if size > 0 else ''}]" for size in sizes) + '; \n'
                    inner_structures.update(field_inner_structures)
                return code, inner_structures, []
        assert isinstance(type_definition, _vk.GTensorMeta), f'Unknown type definition {type_definition}'
        return type_definition.__name__, {}, []  # vec and mats

    @classmethod
    def create_support_code(cls):
        # Gets vulkan device used
        caps = _vk.support()
        code = ""
        if caps.ray_query:
            code += "#define SUPPORTED_RAY_QUERY\n"
        if caps.atom_float:
            code += "#define SUPPORTED_FLOAT_ATOM_ADD\n"
        return code

    @classmethod
    def create_gpu_image(cls, resolution: _typing.Tuple[int, ...], components: int) -> int:
        assert components in [1, 2, 3, 4], "Only 1 to 4 components are supported for images"
        format = [_vk.Format.FLOAT, _vk.Format.VEC2, _vk.Format.VEC3, _vk.Format.VEC4][components - 1]
        dim = len(resolution)
        if dim == 3:
            image = _vk.image_3D(format, resolution[2], resolution[1], resolution[0], mips=1, layers=1)
        elif dim == 2:
            image = _vk.image_2D(format, resolution[1], resolution[0], mips=1, layers=1)
        elif dim == 1:
            image = _vk.image_1D(format, resolution[0], mips=1, layers=1)
        else:
            raise NotImplementedError(f"Unsupported image dimension {dim}")
        l = cls.__IMAGES__[dim - 1]
        r = cls.__IMAGES_REUSABLE_INDICES__[dim - 1]
        if len(r) > 0:
            index = r.pop()
            l[index] = image
        else:
            index = len(l)
            l.append(image)
        return index

    @classmethod
    def upload_image_data(cls, image_dim: int, image_id: int, t: _torch.Tensor):
        im : _vk.Image = cls.__IMAGES__[image_dim - 1][image_id]
        im.load(t)

    @classmethod
    def destroy_gpu_image(cls, image_dim: int, id: int):
        assert image_dim in [1, 2, 3], "Only 1 to 3 dimensions are supported for images"
        l = cls.__IMAGES__[image_dim - 1]
        r = cls.__IMAGES_REUSABLE_INDICES__[image_dim - 1]
        assert id < len(l) and l[id] is not None, f"Trying to destroy a non existing image with id {id} and dimension {image_dim}"
        l[id] = None  # remove reference for automatic destroying
        r.append(id)

    @classmethod
    def dispatch(cls, instance: 'Compute', task: 'ComputeTask'):
        static_signature = (task.rdv_signature, task.rdv_group_size)
        pipeline, ds, images_ds = cls.__EVAL_PIPELINES__.get(static_signature, (None, None, None))
        if pipeline is None: # Create Pipeline if no exist
            pipeline = _vk.pipeline_compute()
            kernel_codes, kernel_dirs = cls.generate_map_kernels(task.binder.dependences)
            compute_code = cls.generate_compute_kernel(instance, task)
            stochastic_flag = "#define RDV_STOCHASTIC_COMPUTE\n" if instance.rdv_stochastic or any(m.is_stochastic for m in task.binder._references()) else ""
            code = f"""
#version 460
#extension GL_GOOGLE_include_directive : require
{
cls.create_support_code()
}
{stochastic_flag}
#define LOCAL_SIZE_X {task.rdv_group_size[0]}
#define LOCAL_SIZE_Y {task.rdv_group_size[1]}
#define LOCAL_SIZE_Z {task.rdv_group_size[2]}
\n""" + __COMPUTE_TEMPLATE__ + "\n" + kernel_codes + "\n" + compute_code
            pipeline.load_shader_from_source(code, include_dirs=set([__INCLUDE_PATH__]+instance.rdv_include_dirs + list(kernel_dirs)))
            # set 0
            pipeline.layout(set=0, binding=0, system_buffer=_vk.DescriptorType.UNIFORM_BUFFER)
            pipeline.layout(set=0, binding=1, deferred_buffer=_vk.DescriptorType.UNIFORM_BUFFER)
            pipeline.layout(set=0, binding=2, parameters_buffer=_vk.DescriptorType.UNIFORM_BUFFER)
            pipeline.layout(set=0, binding=3, rdv_samplers=_vk.DescriptorType.SAMPLER, array_size=2)  # bindless array of samplers images for maps
            # set 1
            pipeline.layout(set=1, binding=4, rdv_textures_1D=_vk.DescriptorType.SAMPLED_IMAGE, array_size=1024)  # bindless array of sampled images for maps
            pipeline.layout(set=1, binding=5, rdv_textures_2D=_vk.DescriptorType.SAMPLED_IMAGE, array_size=1024)  # bindless array of sampled images for maps
            pipeline.layout(set=1, binding=6, rdv_textures_3D=_vk.DescriptorType.SAMPLED_IMAGE, array_size=1024)  # bindless array of sampled images for maps
            pipeline.close()
            ds = pipeline.create_descriptor_set_collection(0, 1)
            # bind system buffer and parameters buffer
            ds[0].update(
                system_buffer=cls.__SYSTEM_BUFFER__,
                deferred_buffer=_DeferredParametersManager.rdv_named_tensors_buffer,
                parameters_buffer=task.rdv_buffer,
                rdv_samplers=[
                    _vk.sampler(),
                    _vk.sampler_linear(address_U=_vk.AddressMode.CLAMP_EDGE, address_V=_vk.AddressMode.CLAMP_EDGE, address_W=_vk.AddressMode.CLAMP_EDGE)
                ]
            )
            images_ds = pipeline.create_descriptor_set_collection(set=1, count=1)
            cls.__EVAL_PIPELINES__[static_signature] = (pipeline, ds, images_ds)
        # binding images
        images_ds[0].update(rdv_textures_1D=cls.__IMAGES__[0])
        images_ds[0].update(rdv_textures_2D=cls.__IMAGES__[1])
        images_ds[0].update(rdv_textures_3D=cls.__IMAGES__[2])
        # create manager if no exist
        dispatch_signature = (static_signature, task.rdv_batches)
        manager = cls.__EVAL_MANAGERS__.get(dispatch_signature)
        if manager is None:  # Create manager
            manager = _vk.compute_manager()
            manager.set_pipeline(pipeline)
            manager.bind(ds[0])
            manager.bind(images_ds[0])
            manager.dispatch_threads(*task.rdv_threads, *task.rdv_group_size)
            manager.freeze()
            cls.__EVAL_MANAGERS__[dispatch_signature] = manager
        r = _generate_seeds()
        t = task.rdv_threads
        with cls.__SYSTEM_BUFFER__ as b:
            b.seeds_x = r[0]
            b.seeds_y = r[1]
            b.seeds_z = r[2]
            b.seeds_w = r[3]
            b.dim_x = t[0]
            b.dim_y = t[1]
            b.dim_z = t[2]

        batches_size = task.rdv_batches
        batches_count = (t[0] - 1) // batches_size[0] + 1, (t[1] - 1) // batches_size[1] + 1, (t[2] - 1) // batches_size[2] + 1

        for bz in range(batches_count[2]):
            for by in range(batches_count[1]):
                for bx in range(batches_count[0]):
                    with cls.__SYSTEM_BUFFER__ as b:
                        b.start_x = bx * batches_size[0]
                        b.start_y = by * batches_size[1]
                        b.start_z = bz * batches_size[2]
                    _vk.submit(manager)

    @classmethod
    def start_session(cls):
        # Define system buffers
        cls.__SYSTEM_BUFFER__ = _vk.object_buffer(layout=_vk.Layout.from_structure(_vk.LayoutAlignment.STD430,
                                                                                    seeds_x=int,
                                                                                    seeds_y=int,
                                                                                    seeds_z=int,
                                                                                    seeds_w=int,
                                                                                    dim_x=int,
                                                                                    dim_y=int,
                                                                                    dim_z=int,
                                                                                    start_x=int,
                                                                                    start_y=int,
                                                                                    start_z=int
                                                                                ), memory=_vk.MemoryLocation.CPU)
        #initialize deferred buffers
        _DeferredParametersManager.init()
        # Defined structs in common.h
        _, inner_structs, _ = cls.create_code_type_definition(MeshInfo, {})
        cls.__DEFINED_STRUCTS__.update(inner_structs)
        _, inner_structs, _ = cls.create_code_type_definition(RaycastableInfo, {})
        cls.__DEFINED_STRUCTS__.update(inner_structs)
        # Add to builtin array to check and dont redefine
        cls.__BUILTIN_STRUCTS__.extend(cls.__DEFINED_STRUCTS__)


def _start_session():
    try:
        __devices = _os.environ['CUDA_VISIBLE_DEVICES'].split(',')  # = str(rdv_device)
        rdv_device = int(__devices[0])
    except:
        rdv_device = 0
    debug = bool(_os.environ.get('RDV_DEBUG', 'False') == 'True')
    _vk.create_device(device=rdv_device, debug=debug)

    if _torch.cuda.is_available():
        _torch.cuda.init()
    _DispatcherEngine.start_session()


class _ComputeMeta(type):
    __COMPUTE_TYPE_COUNTER__ = 0
    """
    Incremental Id for compute types
    """
    __COMPUTE_DYNAMICS_COUNTER__ = 0
    """
    Incremental Id for dynamic computes.
    """

    def __new__(cls, name, bases, dct):
        # Compute type creation
        compute_type = super().__new__(cls, name, bases, dct)
        # Check __extension_info__
        assert '__extension_info__' in dct, 'Derived computes requires a dict __extension_info__ with path or code, parameters, [opt] generics, [opt] include_dirs'
        extension_info = dct['__extension_info__']
        if extension_info is not None:  # is not an abstract node
            extension_path = extension_info.get('path', None)
            extension_code = extension_info.get('code', None)
            stochastic = extension_info.get('stochastic', False)
            pre_eval = extension_info.get('pre_eval', False)
            extension_generics = extension_info.get('generics', {})
            parameters = extension_info.get('parameters', {})
            assert (extension_path is None or isinstance(extension_path, str) and _os.path.isfile(
                extension_path)), 'path must be a valid file path str'
            include_dirs = extension_info.get('include_dirs', [])
            assert (extension_path is None) != (extension_code is None), 'Either path or code must be provided'
            if extension_path is not None:
                include_dirs.append(_os.path.dirname(extension_path))
                extension_code = f"#include \"{_os.path.basename(extension_path)}\"\n"
            extension_dynamic_requires = extension_info.get('dynamics',
                                                            [])  # List with list of map signatures that can be dispatched dynamically by this map
            if len(extension_dynamic_requires) == 0:
                compute_type.rdv_generics = extension_generics
            else:
                _ComputeMeta.__COMPUTE_DYNAMICS_COUNTER__ += 1
                compute_type.rdv_generics = { **extension_generics, 'RDV_DYNAMIC_ID': _MapMeta.__COMPUTE_DYNAMICS_COUNTER__}
            compute_type.rdv_dynamic_requires = extension_dynamic_requires
            compute_object = {'rdv_kernel_id': int, 'rdv_map_pad0': int, 'rdv_map_pad1': int, 'rdv_map_pad2': int,
                          **parameters}
            def from_type_2_layout_description(p, dynamic_array_size=0, **generics):
                if p == Map:
                    return _torch.int64
                if p == _torch.Tensor:
                    return _torch.int64
                if isinstance(p, list):
                    if isinstance(p[0], str):
                        array_size = generics[p[0]]  # get the size from generics
                    else:
                        array_size = p[0] if p[0] > 0 else dynamic_array_size
                    return [array_size, from_type_2_layout_description(p[1], dynamic_array_size, **generics)]
                if isinstance(p, dict):
                    return {'__name__': p.get('__name__'),
                            **{k: from_type_2_layout_description(v, dynamic_array_size, **generics) for k, v in p.items() if
                               k != '__name__'}}
                return p
            compute_object_layout_builder = lambda s, g: _vk.Layout.from_description(
                _vk.LayoutAlignment.SCALAR,
                description=from_type_2_layout_description(compute_object, s, **g)
            )
            compute_type.rdv_stochastic = stochastic
            compute_type.rdv_pre_eval = pre_eval
            compute_type.rdv_layout_builder = compute_object_layout_builder
            compute_type.rdv_type_definition = compute_object
            compute_type.rdv_source_code = extension_code
            compute_type.rdv_include_dirs = include_dirs
            cls.__COMPUTE_TYPE_COUNTER__ += 1
            compute_type.rdv_type_id = cls.__COMPUTE_TYPE_COUNTER__
        return compute_type


class _MapMeta(_ComputeMeta):
    def __call__(self, *args, **kwargs):
        # map instantiation
        map_instance: Map = super(_MapMeta, self).__call__(*args, **kwargs)

        param_requires_grad_generic = {'PARAMS_REQUIRES_GRAD': 1} if bool(map_instance.deferred_info) else {}
        stochastic_generic = {'STOCHASTIC': 1} if map_instance.is_stochastic else {}
         # update generics with param_requires_grad and stochastic
        map_instance.rdv_generics = {**map_instance.rdv_generics, **param_requires_grad_generic, **stochastic_generic}
        if not map_instance.is_generic:
            assert all(not m.is_generic for m in map_instance.children), f'A non-generic map {type(map_instance)} can not contains generic submaps'
            compute_id = _DispatcherEngine.register_instance(map_instance)
            map_instance.rdv_kernel_id = compute_id
            map_instance.rdv_buffer_accessor.rdv_kernel_id = compute_id  # set the id to the gpu.
        return map_instance


class MapElement:
    def __init__(self,  type_definition, accessor, generics):
        object.__setattr__(self, '_type_definition', type_definition)
        object.__setattr__(self, '_generics', generics)
        object.__setattr__(self, '_accessor', accessor)
        object.__setattr__(self, '_deferreds_cache', {})

    def _compute_ids(self, deep=False) -> _typing.Iterable[int]:
        pass

    @cached_property
    def dependences(self):
        return set(self._compute_ids(True))

    def _deferreds(self) -> _typing.Dict[str, int]:
        """
        Returns a dict of deferred parameters used in this element.
        key: name of the deferred parameter
        value: dimension required for the tensor
        """
        return object.__getattribute__(self, '_deferreds_cache')

    def _references(self) -> _typing.Iterable['Map']:
        pass


class MapArray(MapElement):
    def __init__(self, type_definition, accessor, generics):
        super().__init__(type_definition, accessor, generics)
        object.__setattr__(self, '_element_definition', type_definition[1])
        if isinstance(type_definition[0], str):
            size = generics[type_definition[0]]
        else:
            assert isinstance(type_definition[0], int)
            size = type_definition[0]
        object.__setattr__(self, '_backend_array', [None] * size)

    def _compute_ids(self, deep=False):
        element_definition = object.__getattribute__(self, '_element_definition')
        backend_array = object.__getattribute__(self, '_backend_array')
        if element_definition == Map:
            first_map: _typing.Optional[Map] = next((x for x in backend_array if x is not None), None)
            assert first_map is not None and all(m.rdv_kernel_id == first_map.rdv_kernel_id for m in backend_array if m is not None)
            return (first_map.rdv_kernel_id, *first_map.extended_dependencies(), *(() if not deep else first_map.rdv_parameters._compute_ids(deep=True)))
        if isinstance(element_definition, dict) or isinstance(element_definition, list):
            first_element : _typing.Optional[MapElement] = backend_array[0]
            if first_element is not None:
                cids = first_element._compute_ids(deep)
            else:
                cids = ()
            ## TODO: check this in assignment
            # assert first_element is not None and all(m._compute_ids() == cids for m in backend_array)
            return cids # if not deep else first_element._compute_ids(deep=True)
        return ()

    def _references(self):
        element_definition = object.__getattribute__(self, '_element_definition')
        backend_array = object.__getattribute__(self, '_backend_array')
        if element_definition == Map:
            return set(backend_array)
        if element_definition == _torch.int64:
            return set(m for m in backend_array if isinstance(m, Map))  # collect only maps
        references = set()
        if isinstance(element_definition, dict) or isinstance(element_definition, list):
            for m in backend_array:
                if m is not None:
                    references.update(m._references())
        return references

    def __len__(self):
        backend_array = object.__getattribute__(self, '_backend_array')
        return len(backend_array)

    def __getitem__(self, item):
        element_definition = object.__getattribute__(self, '_element_definition')
        backend_array = object.__getattribute__(self, '_backend_array')
        accessor = object.__getattribute__(self, '_accessor')
        generics = object.__getattribute__(self, '_generics')
        if item < 0 or item >= len(backend_array):
            raise IndexError("Index out of range")
        if backend_array[item] is not None:
            return backend_array[item]
        if isinstance(element_definition, list):  # subarray
            value = MapArray(element_definition, accessor[item] if accessor else None, generics)
        elif isinstance(element_definition, dict):  # substructure
            value = MapStruct(element_definition, accessor[item] if accessor else None, generics)
        elif element_definition == Map or element_definition == _torch.int64:
            value = None
        else:
            value = element_definition()
        backend_array[item] = value
        return value

    def __setitem__(self, key, value):
        element_definition = object.__getattribute__(self, '_element_definition')
        backend_array = object.__getattribute__(self, '_backend_array')
        accessor = object.__getattribute__(self, '_accessor')
        deferreds_cache = object.__getattribute__(self, '_deferreds_cache')
        assert key >= 0 and key < len(backend_array)
        is_deferrable = element_definition == _torch.Tensor and isinstance(value, deferred)
        assert not isinstance(element_definition, list) and not isinstance(element_definition, dict)
        if is_deferrable:
            # Collect deferred parameters
            if value._key in deferreds_cache:
                assert deferreds_cache[value._key] == value.dimension, f"Deferred parameter {value._key} already assigned requiring different dimension {self._deferreds_cache[value._key]} vs {value.dimension}"
            deferreds_cache[value._key] = value.dimension
        if accessor is not None:
            accessor_value = value if not is_deferrable else value.as_uint64
            if element_definition == _torch.int64 or element_definition == _torch.Tensor or element_definition == Map:
                if not isinstance(value, _vk.GPUPtr):
                    accessor_value = _vk.wrap_gpu(accessor_value, 'in')
            accessor[key] = accessor_value
        backend_array[key] = value


class MapStruct(MapElement):
    def __init__(self, type_definition, accessor, generics):
        super().__init__(type_definition, accessor, generics)

    def __getattr__(self, item):
        type_definition = object.__getattribute__(self, '_type_definition')
        accessor = object.__getattribute__(self, '_accessor')
        generics = object.__getattribute__(self, '_generics')
        assert item in type_definition
        try:
            return super().__getattribute__(item)
        except:
            pass
        if item == '__name__':
            return type_definition['__name__']
        field_type = type_definition[item]
        if isinstance(field_type, dict):  # sub-structure
            field_value = MapStruct(field_type, getattr(accessor, item) if accessor is not None else None, generics)
        elif isinstance(field_type, list):
            field_value = MapArray(field_type, getattr(accessor, item) if accessor is not None else None, generics)
        elif field_type == Map or field_type == _torch.int64:
            field_value = None
        else:
            field_value = field_type()
        object.__setattr__(self, item, field_value)
        return field_value

    def __setattr__(self, key, value):
        type_definition = object.__getattribute__(self, '_type_definition')
        accessor = object.__getattribute__(self, '_accessor')
        deferreds_cache = object.__getattribute__(self, '_deferreds_cache')
        assert key in type_definition
        field_definition = type_definition[key]
        is_deferrable = field_definition == _torch.Tensor and isinstance(value, deferred)
        assert not isinstance(field_definition, list), "Can not set directly a list, use per-index access"
        assert not isinstance(field_definition, dict), "Can not set directly an struct, use per-field access"
        if is_deferrable:
            # Collect deferred parameters
            if value._key in deferreds_cache:
                assert deferreds_cache[value._key] == value.dimension, f"Deferred parameter {value._key} already assigned requiring different dimension {self._deferreds_cache[value._key]} vs {value.dimension}"
            deferreds_cache[value._key] = value.dimension
        if accessor is not None:
            assert not isinstance(value, Map) or not value.is_generic, "Can not assign a generic submap to a non-generic map. Submap should be castable to a non-generic version of the map."
            accessor_value = value if not is_deferrable else value.as_uint64
            if field_definition == _torch.int64 or field_definition == _torch.Tensor or field_definition == Map:
                if not isinstance(value, _vk.GPUPtr):
                    accessor_value = _vk.wrap_gpu(accessor_value, 'in')
            setattr(accessor, key, accessor_value)
        super().__setattr__(key, value)

    def _compute_ids(self, deep=False):
        type_definition = object.__getattribute__(self, '_type_definition')
        cids = []
        for key in type_definition:
            field_type = type_definition[key]
            field_value = getattr(self, key)
            if field_type == Map:
                assert field_value is not None
                cids.append(field_value.rdv_kernel_id)
                cids.extend(field_value.extended_dependencies())
                if deep:
                    cids.extend(field_value.rdv_parameters._compute_ids(True))
            elif isinstance(field_type, dict) or isinstance(field_type, list):
                cids.extend(field_value._compute_ids(deep))
        return tuple(cids)

    def _references(self):
        type_definition = object.__getattribute__(self, '_type_definition')
        references = set()
        for key in type_definition:
            field_type = type_definition[key]
            field_value = getattr(self, key)
            if field_type == Map or field_type == _torch.int64 and isinstance(field_value, Map):
                assert field_value is not None, f"Field {key} was expected to be a Map reference but is None."
                references.add(field_value)
            elif isinstance(field_type, dict) or isinstance(field_type, list):
                if field_value is None:
                    continue
                references.update(field_value._references())
        return references


class Map(object, metaclass=_MapMeta):
    """
    Base class for all maps.
    A map defines a transform from R^n to R^m.
    Forward and backward operations are solved with a compute shader.
    """

    __extension_info__ = None  # Represent abstract nodes with __extension_info__ None
    rdv_include_dirs = []
    rdv_generics = {}
    """
    Gets the set of generics used by this map. All generics are turn into defines in the code that are
    only valid within the map implementation.
    """
    rdv_stochastic = False
    """
    Indicates if the map uses stochastic operations. 
    Forward and backward passes will start with the same random seed.
    """
    rdv_dynamic_requires = {}
    """
    Gets the dynamic signatures (input_dim, output_dim) required by this map dynamic accesses.
    """
    rdv_type_id = 0
    """
    Each Map derived class has a unique type_id.
    """
    rdv_layout_builder = None
    """
    A function that receives a final count and retrieves the layout for the object buffer creation.
    """
    rdv_buffer = None
    """
    Vulky uniform buffer with the struct defining the parameters of the map. 
    """
    rdv_buffer_accessor = None
    """
    Object Buffer Accessor to object header and all map parameters.
    """
    rdv_parameters = None
    """
    MapStruct with buffer accessor to the defining the parameters of the map. 
    """
    rdv_type_definition = None
    """
    Dictionary with the definition of parameters of the map.
    """
    rdv_source_code = None
    """
    Code for the specific map kernel.
    """
    rdv_frozen = False
    """
    Once the map is initialized it is frozen to new updates of the parameters.
    """
    def __init__(self,
                 *,
                 dynamic_length=0,
                 input_dim=None,
                 output_dim=None,
                 input_requires_grad=False,
                 bw_uses_output=False,
                 **generics):
        input_requires_grad_generic = { 'INPUT_REQUIRES_GRAD': 1 } if input_requires_grad else {}
        bw_uses_output_generic = { 'BW_USES_OUTPUT': 1 } if bw_uses_output else {}
        self.rdv_generics = {
            **self.rdv_generics,
            **{k: v for k,v in generics.items() if v is not None},
            **input_requires_grad_generic,
            **bw_uses_output_generic,
            'INPUT_DIM': input_dim, 'OUTPUT_DIM': output_dim
        }
        if not self.is_generic:
            layout = type(self).rdv_layout_builder(dynamic_length, self.rdv_generics)
            buffer = _vk.object_buffer(layout).clear()
            object.__setattr__(self, 'rdv_buffer', buffer)
            object.__setattr__(self, 'rdv_buffer_accessor', buffer.accessor)
            object.__setattr__(self, 'rdv_parameters', MapStruct(self.rdv_type_definition, buffer.accessor, self.rdv_generics))
        else:
            object.__setattr__(self, 'rdv_parameters', MapStruct(self.rdv_type_definition, None, self.rdv_generics))
        object.__setattr__(self, 'rdv_cast_cache', None)

    @cached_property
    def deferred_info(self) -> _typing.Dict[str, int]:
        """
        Returns a dict of deferred parameters used in this map and its children.
        key: name of the deferred parameter
        value: dimension required for the tensor
        """
        d = self.rdv_parameters._deferreds()
        for c in self.children:
            for k, v in c.deferred_info.items():
                if k in d:
                    assert v == d[k], f'Deferred parameter {k} has different required dimensions in map or child map.'
                else:
                    d[k] = v
        return d

    @cached_property
    def is_stochastic(self) -> bool:
        """
        Indicates if the map or any of its children use stochastic operations.
        """
        return self.rdv_stochastic or any(child.is_stochastic for child in self.children)

    @cached_property
    def input_requires_grad(self) -> bool:
        return 'INPUT_REQUIRES_GRAD' in self.rdv_generics

    @cached_property
    def bw_uses_output(self) -> bool:
        return 'BW_USES_OUTPUT' in self.rdv_generics

    # @cached_property
    # def requires_pre_eval(self) -> bool:
    #     """
    #     Indicates if the map or any of its children require pre-evaluation.
    #     """
    #     return self.rdv_pre_eval or any(child.requires_pre_eval for child in self.children)
    #
    # def pre_eval(self, **deferred_tensors: _torch.Tensor):
    #     for c in self.children:
    #         if c.requires_pre_eval:
    #             c.pre_eval(**deferred_tensors)

    def clone(self,
              **kwargs) -> 'Map':
        raise NotImplementedError()

    def cast(self,
             input_dim=None,
             output_dim=None,
             input_requires_grad=None,
             bw_uses_output=None) -> 'Map':
        if self.rdv_cast_cache is None:
            object.__setattr__(self, 'rdv_cast_cache', {})
        key = (input_dim, output_dim, input_requires_grad, bw_uses_output)
        if key in self.rdv_cast_cache:
            weak_ref = self.rdv_cast_cache[key]
            casted = weak_ref()
            if casted is not None:
                return casted
        changed = False
        promoting = None
        if input_dim is None:
            input_dim = self.input_dim
        if self.input_dim != input_dim:
            assert self.input_dim is None
            changed |= True
        if output_dim is None:
            output_dim = self.output_dim
        if output_dim != self.output_dim:
            assert self.output_dim is None or self.output_dim == 1, f"Mismatch in output dim casting to {output_dim} having {self.output_dim}."
            if self.output_dim == 1:
                promoting = output_dim
                output_dim = 1  # keeps output dim 1 but perform a promote
                changed |= False
            else:
                changed |= True
        if input_requires_grad is None:
            input_requires_grad = self.input_requires_grad
        changed |= input_requires_grad != self.input_requires_grad
        if bw_uses_output is None:
            bw_uses_output = self.bw_uses_output
        changed |= bw_uses_output != self.bw_uses_output
        s = self
        if changed:
            s = self.clone(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        if promoting is not None:
            s = s.promote(promoting)
        self.rdv_cast_cache[key] = _weakref.ref(s)
        object.__setattr__(s, 'rdv_cast_cache', self.rdv_cast_cache)  # share cache
        return s

    @cached_property
    def device_ptr(self):
        return self.rdv_buffer.device_ptr

    @cached_property
    def _compute_ids(self):
        return self.rdv_parameters._compute_ids()

    @cached_property
    def rdv_signature(self):
        assert not self.is_generic, "Signatures represent uniquely a computation unit. Generic maps do not have signatures."
        return (self.rdv_type_id, frozenset(self.rdv_generics.items()), self.rdv_parameters._compute_ids())

    def __getattr__(self, item):
        if item in self.rdv_type_definition:
            return getattr(self.rdv_parameters, item)
        return super().__getattribute__(item)

    def __setattr__(self, key, value):
        if key in self.rdv_type_definition:
            assert not self.rdv_frozen, "Parameters of a map can only be set during init."
            return setattr(self.rdv_parameters, key, value)
        super().__setattr__(key, value)

    def __getitem__(self, item):
        if isinstance(item, int):
            return SelectIndexesMap(self, [item])
        if isinstance(item, tuple) or isinstance(item, list):
            return SelectIndexesMap(self, list(item))
        if isinstance(item, slice):
            if self.output_dim is None:
                indices = list(range(10000)) # sufficently large to index
            else:
                indices = [i for i in range(self.output_dim)]
            return SelectIndexesMap(self, indices[item])
        raise Exception(f"Not supported index/slice object {type(item)}")


    @cached_property
    def input_dim(self):
        """
        Gets the dimension of the input vector.
        If None, the map is generic in the input and should be cast to have a non-generic map.
        """
        return self.rdv_generics.get('INPUT_DIM')

    @cached_property
    def output_dim(self):
        """
        Gets the dimension of the output vector.
        If None, the map is generic in the output and should be cast to have a non-generic map.
        """
        return self.rdv_generics.get('OUTPUT_DIM')

    @cached_property
    def is_generic(self):
        return self.input_dim is None or self.output_dim is None

    @cached_property
    def is_generic_input(self):
        return self.input_dim is None

    @cached_property
    def is_generic_output(self):
        return self.output_dim is None

    @cached_property
    def is_dynamic(self):
        return len(self.rdv_dynamic_requires) > 0

    def extended_dependencies(self) -> _typing.Set[int]:
        """
        Returns this map's extra dependencies compute ids as a set.
        """
        return set()

    @cached_property
    def children(self) -> _typing.Iterable['Map']:
        """
        Returns an iterable for all accessible direct submaps.
        """
        return self.rdv_parameters._references()

    def domain_range(self, xmin: TensorLike, xmax: TensorLike) -> 'Map':
        return DomainRangeMap(self, xmin, xmax, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def domain_mask(self, mask: MapLike) -> 'Map':
        return DomainMaskMap(self, as_map(mask), input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def __call__(self, *args, **kwargs):
        input, = args
        for k, v in self.deferred_info.items():
            assert k in kwargs, f"Missing deferred parameter {k} required by the map."
            assert len(kwargs[k].shape) == v, f"Deferred parameter {k} must have dimension {v} as required by the map."
        names = kwargs.keys()
        deferred_tensors = kwargs.values()
        assert len(self.deferred_info) == len(names) == len(deferred_tensors), "Deferred parameters must match the deferred parameters required by the map."
        return _MapEvalFunction.apply(self, input, names, *deferred_tensors)

    def __add__(self, other: 'MapLike') -> 'Map':
        return AdditionMap(self, other, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def __radd__(self, other: 'MapLike') -> 'Map':
        return AdditionMap(other, self, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def __sub__(self, other: 'MapLike') -> 'Map':
        return SubtractionMap(self, other, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def __rsub__(self, other: 'MapLike') -> 'Map':
        return SubtractionMap(other, self, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def __mul__(self, other: 'MapLike') -> 'Map':
        return MultiplicationMap(self, other, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def __rmul__(self, other: 'MapLike') -> 'Map':
        return MultiplicationMap(other, self, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def __truediv__(self, other: 'MapLike') -> 'Map':
        return DivisionMap(self, other, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def __rtruediv__(self, other: 'MapLike') -> 'Map':
        return DivisionMap(other, self, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def __or__(self, other):
        return ConcatMap(self, other, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def __ror__(self, other):
        return ConcatMap(other, self, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def promote(self, dim: int):
        return PromotedMap(
            map=self,
            input_dim=self.input_dim,
            output_dim=dim,
            input_requires_grad=self.input_requires_grad,
            bw_uses_output=self.bw_uses_output
        )

    def then(self, outer: 'MapLike') -> 'Map':
        outer = as_map(outer)
        if isinstance(self, IdentityMap):
            return outer.cast(input_dim=self.output_dim, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)
        if isinstance(outer, IdentityMap):
            return self.cast(output_dim=outer.output_dim)
        return ComposeMap(self, outer, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def after(self, inner: 'MapLike') -> 'Map':
        inner = as_map(inner)
        if isinstance(self, IdentityMap):
            return inner.cast(output_dim=self.input_dim, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)
        if isinstance(inner, IdentityMap):
            return self.cast(input_dim=inner.input_dim)
        return ComposeMap(inner, self, input_requires_grad=self.input_requires_grad, bw_uses_output=self.bw_uses_output)

    def capture(
            self,
            *shape: int,
            sample_location: tuple | SampleLocation = SampleLocation.CENTER,
            probes_map: _typing.Optional['Map'] = None,
    **deferred_parameters: _torch.Tensor) -> _torch.Tensor:
        return Sensor(
            *shape,
            samples_location=sample_location,
            probes_map=probes_map
        ).view(self).capture(**deferred_parameters)

    def backward(self,
                 input: _torch.Tensor,
                 output_grad: _torch.Tensor,
                 parameters: _typing.Dict[str, _typing.Any],
                 parameters_grad: _typing.Dict[str, _typing.Any]):
        pass


class ComputeTask:
    def __init__(self, signature: tuple, buffer: _vk.ObjectBuffer, map_object: MapStruct,
                 threads: tuple,
                 batches: tuple,
                 group_size: tuple = (32, 1, 1),
                 **generics):
        self.rdv_signature = signature
        self.rdv_threads = threads
        self.rdv_batches = batches
        self.rdv_group_size = group_size
        self.rdv_buffer = buffer
        self.rdv_object = map_object
        self.rdv_generics = generics

    def save(self, *values):
        self.saved_values = values

    @cached_property
    def binder(self) -> MapStruct:
        return self.rdv_object


class Compute(Singleton, metaclass=_ComputeMeta):
    """
    Base class for compute-based tensor operations in rendervous.
    Derived classes must define two stages: bind and result.
    __extension_info__ must provide parameters: dict and code|path with a MAIN(tid) { } method.
    generics in extension info can be used to define sizes and parameters needed as compile-time constant in the code.
    """
    __extension_info__ = None  # Abstract node
    __OBJECTS__ = { }  # from signature to binders
    __LOCK__ = _threading.Lock()
    __CAST__ = _weakref.WeakKeyDictionary()

    def cast_map(self, map: 'Map', input_dim=None, output_dim=None, input_requires_grad=None, bw_uses_output=None) -> 'Map':
        key = (input_dim, output_dim, input_requires_grad, bw_uses_output)
        if map not in self.__CAST__:
            self.__CAST__[map] = {}
        if key not in self.__CAST__[map]:
            self.__CAST__[map][key] = map.cast(
                input_dim=input_dim,
                output_dim=output_dim,
                input_requires_grad=input_requires_grad,
                bw_uses_output=bw_uses_output
            )
        return self.__CAST__[map][key]

    @classmethod
    def eval(cls, *args, deferred_parameters: _typing.Optional[_typing.Dict[str, _torch.Tensor]] = None, compute_grads: bool = False, **kwargs):
        #with cls.__LOCK__:
        instance = cls.get_instance()
        compute_task = instance.bind(*args, **kwargs)
        compute_task.binder._references()
        if deferred_parameters is not None:
            grads = _DeferredParametersManager.bind(deferred_parameters, compute_grads)
        else:
            grads = {}
        _DispatcherEngine.dispatch(instance, compute_task)
        if deferred_parameters is not None:
            _DeferredParametersManager.unbind(deferred_parameters, grads=compute_grads)
        r = instance.result(compute_task)
        if compute_grads:
            return r, grads
        return r

    @classmethod
    def create_task(cls,
                    threads: int | tuple,
                    *maps: Map,
                    dynamic_size=0,
                    batches: None | int | tuple = None,
                    group_size: tuple = (32, 1, 1),
                    **generics) -> ComputeTask:
        generics = cls.rdv_generics if len(generics) == 0 else { **cls.rdv_generics, **generics }
        if any(len(m.deferred_info) > 0 for m in maps):
            generics.update(RDV_HAS_DEFERRED=1)
        if isinstance(threads, int):
            threads = (threads, 1, 1)
        elif len(threads) == 2:
            threads = (threads[0], threads[1], 1)
        else:
            assert len(threads) == 3
        if batches is None:
            batches = threads
        elif isinstance(batches, int):
            batches = (batches, 1, 1)
        elif len(batches) == 2:
            batches = (batches[0], batches[1], 1)
        else:
            batches = tuple(batches)
        signature = (cls.rdv_type_id, *(m.rdv_kernel_id if m else 0 for m in maps), frozenset(generics.items()))
        obj, map_struct = cls.__OBJECTS__.get((signature, dynamic_size), (None, None))
        if obj is None:
            layout = cls.rdv_layout_builder(dynamic_size, generics)
            obj = _vk.object_buffer(layout, memory=_vk.MemoryLocation.CPU)
            map_struct = MapStruct(cls.rdv_type_definition, obj.accessor, generics)
            cls.__OBJECTS__[(signature, dynamic_size)] = (obj, map_struct)
        return ComputeTask(signature, obj, map_struct, threads, batches, group_size, **generics)

    def bind(self, *args, **kwargs) -> 'ComputeTask':
        '''
        sets the arguments to the object and return the number of threads to dispatch
        '''
        raise NotImplementedError()

    def result(self, compute_task) -> _typing.Any:
        '''
        :param compute_task: task being computed. Access the binder to get the output tensors.
        :return: resultant output, can be directly tensors or a postprocessing on them.
        '''
        raise NotImplementedError()


class _MapForwardEvalCompute(Compute):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/system/map_forward_eval.h',
        parameters=dict(
            input_tensor=_torch.Tensor,
            output_tensor=_torch.Tensor,
            map=Map
        )
    )

    def bind(self, *args, **kwargs) -> ComputeTask:
        map, input = args
        input_dim = 0 if isinstance(input, int) else input.shape[-1]
        map = self.cast_map(
            map,
            input_dim=input_dim,
            input_requires_grad=input.requires_grad if isinstance(input, _torch.Tensor) else False,
            bw_uses_output=False
        )
        output_dim = map.output_dim
        assert not map.is_generic
        if isinstance(input, int):
            batch = (input,)
            input = None  # not used in forward eval
        else:
            batch = input.shape[:-1]
        # number of elements to process
        count = input.numel() // input_dim if input is not None else batch[0]
        output = _vk.tensor(*batch, output_dim)
        task = _MapForwardEvalCompute.create_task(count, map, MAP_INPUT_DIM=input_dim, MAP_OUTPUT_DIM=output_dim)
        binder = task.binder
        binder.input_tensor = _vk.wrap_gpu(input, 'in')
        binder.output_tensor = _vk.wrap_gpu(output, 'out')
        binder.map = map
        return task

    def result(self, compute_task: ComputeTask) -> _typing.Any:
        compute_task.binder.input_tensor.unwrap()
        return compute_task.binder.output_tensor.unwrap()


class _MapBackwardEvalCompute(Compute):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/system/map_backward_eval.h',
        parameters=dict(
            input_tensor=_torch.Tensor,
            output_grad_tensor=_torch.Tensor,
            input_grad_tensor=_torch.Tensor,
            map=Map
        )
    )

    def bind(self, *args, **kwargs) -> ComputeTask:
        map, input, output_grad = args
        input_dim = 0 if isinstance(input, int) else input.shape[-1]
        map = self.cast_map(
            map,
            input_dim=input_dim,
            input_requires_grad=input.requires_grad if isinstance(input, _torch.Tensor) else False,
            bw_uses_output=False
        )
        assert not map.is_generic
        output_dim = map.output_dim
        if isinstance(input, int):
            count = input
            input = None
            input_grad = None
        else:
            count = input.numel() // input_dim
            input_grad = None if not input.requires_grad else _vk.zeros(*input.shape[:-1], input_dim)
            assert map.input_dim == input_dim and map.output_dim == map.output_dim
        input_requires_grad_generics = {} if input_grad is None else {'INPUT_REQUIRES_GRAD': 1}
        task = _MapBackwardEvalCompute.create_task(
            count,
            map,
            MAP_INPUT_DIM=input_dim,
            MAP_OUTPUT_DIM=output_dim,
            **input_requires_grad_generics
        )
        binder = task.binder
        binder.input_tensor = _vk.wrap_gpu(input, 'in')
        binder.output_grad_tensor = _vk.wrap_gpu(output_grad, 'in')
        binder.input_grad_tensor = _vk.wrap_gpu(input_grad, 'out')
        binder.map = map
        return task

    def result(self, compute_task: ComputeTask) -> _typing.Any:
        compute_task.binder.input_tensor.unwrap()
        compute_task.binder.output_grad_tensor.unwrap()
        return compute_task.binder.input_grad_tensor.unwrap()


class _MapEvalFunction(_torch.autograd.Function):
    @staticmethod
    def forward(ctx: _typing.Any, *args: _typing.Any, **kwargs: _typing.Any) -> _typing.Any:
        map, input, names, *deferred_tensors = args
        if isinstance(input, _torch.Tensor):
            save_input = (input,)
        else:
            save_input = ()
            assert isinstance(input, int)
            ctx.input_number = input
        ctx.map = map
        ctx.names = names
        ctx.save_for_backward(*save_input, *deferred_tensors)
        output = _MapForwardEvalCompute.eval(map, input, deferred_parameters={n: t for n, t in zip(names, deferred_tensors)}, compute_grads=False)
        return output

    @staticmethod
    def backward(ctx: _typing.Any, *grad_outputs: _typing.Any) -> _typing.Any:
        output_grad, = grad_outputs
        if hasattr(ctx, 'input_number'):
            input = ctx.input_number
            deferred_tensors = ctx.saved_tensors
        else:
            input, *deferred_tensors = ctx.saved_tensors
        map = ctx.map
        names = ctx.names
        input_grad, grads = _MapBackwardEvalCompute.eval(map, input, output_grad, deferred_parameters={n: t for n, t in zip(names, deferred_tensors)}, compute_grads=True)
        return (
            None, # map
            input_grad,
            None, # names
            *(grads.get(k) for k in names)
        )


# Capturing efficiently creates a compute task permanently bound to a specific map and shape.

class _CaptureForwardCompute(Compute):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/system/capture_forward.h',
        stochastic=True,
        parameters=dict(
            probes_map=Map,
            field_map=Map,
            rdv_indices=_torch.Tensor,
            output_tensor=_torch.Tensor,
            shape=['INDEX_DIM', int],
            sample_location=['INDEX_DIM', int],
            samples=int
        )
    )

    def bind(self, *args, **kwargs) -> 'ComputeTask':
        probes_map, field, shape, sampling_modes, indices, samples = args
        index_dim = len(shape)
        probes_map = self.cast_map(
            probes_map,
            input_dim=index_dim,
            input_requires_grad=False,
            bw_uses_output=True
        )
        assert not probes_map.is_generic, "Probe map can not be generic after input_dim is specified."
        field = self.cast_map(
            field,
            input_dim=probes_map.output_dim,
            input_requires_grad=bool(probes_map.deferred_info),
            bw_uses_output=False
        )
        assert not field.is_generic, "Field map can not be generic after input_dim is specified."
        output_dim = field.output_dim
        import math
        count = math.prod(shape)
        task = _CaptureForwardCompute.create_task(
            count, probes_map, field, INDEX_DIM=index_dim, MAP_INPUT_DIM=field.input_dim, MAP_OUTPUT_DIM=output_dim
        )
        output = _vk.tensor(*shape, output_dim)
        binder = task.binder
        binder.rdv_indices = indices
        binder.output_tensor = _vk.wrap_gpu(output, 'out')
        binder.probes_map = probes_map
        binder.field_map = field
        shape_len = len(shape)
        for i, (d, mode) in enumerate(zip(shape, sampling_modes)):
            binder.shape[shape_len - i - 1] = d
            binder.sample_location[shape_len - i - 1] = int(mode)
        binder.samples = samples
        return task

    def result(self, compute_task) -> _typing.Any:
        return compute_task.binder.output_tensor.unwrap()


class _CaptureBackwardCompute(Compute):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__ + '/system/capture_backward.h',
        stochastic=True,
        parameters=dict(
            probes_map=Map,
            field_map=Map,
            rdv_indices=_torch.Tensor,
            output_grad_tensor=_torch.Tensor,
            shape=['INDEX_DIM', int],
            sample_location=['INDEX_DIM', int],
            samples=int
        )
    )

    def bind(self, *args, **kwargs) -> 'ComputeTask':
        probes_map, field, shape, sampling_modes, indices, samples = args
        index_dim = len(shape)
        probes_map = self.cast_map(
            probes_map,
            input_dim=index_dim,
            input_requires_grad=False,
            bw_uses_output=True
        )
        assert not probes_map.is_generic, "Probe map can not be generic after input_dim is specified."
        field = self.cast_map(
            field,
            input_dim=probes_map.output_dim,
            input_requires_grad=bool(probes_map.deferred_info),
            bw_uses_output=False
        )
        assert not field.is_generic, "Field map can not be generic after input_dim is specified."
        output_dim = field.output_dim
        import math
        count = math.prod(shape)
        task = _CaptureBackwardCompute.create_task(
            count, probes_map, field, INDEX_DIM=index_dim, MAP_INPUT_DIM=field.input_dim, MAP_OUTPUT_DIM=output_dim
        )
        binder = task.binder

        binder.rdv_indices = indices
        binder.probes_map = probes_map
        binder.field_map = field
        shape_len = len(shape)
        for i, (d, mode) in enumerate(zip(shape, sampling_modes)):
            binder.shape[shape_len - i - 1] = d
            binder.sample_location[shape_len - i - 1] = int(mode)
        binder.samples = samples
        return task

    def result(self, compute_task) -> _typing.Any:
        # compute_task.binder.output_grad_tensor.unwrap()
        return None


class _SensorFieldCaptureFunction(_torch.autograd.Function):
    @staticmethod
    def forward(ctx: _typing.Any, *args: _typing.Any, **kwargs: _typing.Any) -> _typing.Any:
        sensor_field, indices, names, *deferred_tensors = args
        ctx.sensor_field = sensor_field
        ctx.indices = indices
        ctx.names = names
        ctx.save_for_backward(*deferred_tensors)
        output = sensor_field._forward(indices, **{n: t for n, t in zip(names, deferred_tensors)})
        return output

    @staticmethod
    def backward(ctx: _typing.Any, *grad_outputs: _typing.Any) -> _typing.Any:
        output_grad, = grad_outputs
        sensor_field = ctx.sensor_field
        indices = ctx.indices
        names = ctx.names
        deferred_tensors = ctx.saved_tensors
        grads = sensor_field._backward(
            indices,
            output_grad,
            **{n: t for n, t in zip(names, deferred_tensors)}
        )
        return (
            None,  # sensor_field
            None,  # indices
            None,  # names
            *(grads.get(k) for k in names)
        )


class Sensor:
    def __init__(self,
                 *shape: int,
                 samples_location: _typing.Tuple[SampleLocation,...] | SampleLocation = SampleLocation.CENTER,
                 probes_map: 'MapLike' = None,
                 ):
        self.shape = tuple(shape)
        if isinstance(samples_location, int):
            self.samples_location = (samples_location,) * len(shape)
        else:
            assert len(samples_location) == len(shape)
            self.samples_location = samples_location
        self.probes_map = as_map(
            probes_map,
            default=IdentityMap.get_instance()
        )

    def view(self, field: 'MapLike', samples: int = 1, bw_samples: int | None = None) -> 'SensorView':
        field_map = as_map(field, default=IdentityMap.get_instance())
        return SensorView(
            sensor=self,
            field=field_map,
            samples=samples,
            bw_samples=bw_samples
        )


class SensorView:
    def __init__(self,
                 sensor: Sensor,
                 field: 'MapLike',
                 samples: int = 1,
                 bw_samples: int | None = 1
                 ):
        if bw_samples is None:
            bw_samples = samples
        self.fw_task = _CaptureForwardCompute.get_instance().bind(
            sensor.probes_map,
            field,
            sensor.shape,
            sensor.samples_location,
            deferred('rdv_indices', shape=(-1, len(sensor.shape))),
            samples
        )
        self.bw_task = _CaptureBackwardCompute.get_instance().bind(
            sensor.probes_map,
            field,
            sensor.shape,
            sensor.samples_location,
            deferred('rdv_indices', shape=(-1, len(sensor.shape))),
            bw_samples
        )

    def _forward(self, indices: _torch.Tensor | None, **kwargs) -> _torch.Tensor:
        _DeferredParametersManager.bind({'rdv_indices': indices, **kwargs})
        _DispatcherEngine.dispatch(_CaptureForwardCompute.get_instance(), self.fw_task)
        _DeferredParametersManager.unbind({'rdv_indices': indices, **kwargs}, grads=False)
        return _CaptureForwardCompute.get_instance().result(self.fw_task)

    def _backward(self, indices: _torch.Tensor | None, output_grad: _torch.Tensor, **kwargs) -> _typing.Dict[str, _torch.Tensor]:
        grads = _DeferredParametersManager.bind({'rdv_indices': indices, **kwargs}, compute_grads=True)
        # set output grad here, only bind required per-call
        self.bw_task.binder.output_grad_tensor = _vk.wrap_gpu(output_grad, 'in')
        _DispatcherEngine.dispatch(_CaptureBackwardCompute.get_instance(), self.bw_task)
        self.bw_task.binder.output_grad_tensor.unwrap()
        _DeferredParametersManager.unbind({'rdv_indices': indices, **kwargs}, grads=True)
        _CaptureBackwardCompute.get_instance().result(self.bw_task)
        return grads

    def capture(self, indices: _torch.Tensor | None = None, **kwargs) -> _torch.Tensor:
        return _SensorFieldCaptureFunction.apply(
            self,
            indices,
            tuple(kwargs.keys()),
            *kwargs.values()
        )


# Initialize the dispatcher engine
_start_session()


class ConstantMap(Map):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_const.h",
        parameters=dict(
            t=_torch.Tensor
        )
    )

    def __init__(self, t: TensorLike | deferred, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        t = ensure_tensor(t, map_dim=1)
        if output_dim is None:
            output_dim = t.shape[0]
        assert output_dim == t.shape[0]
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.t = t

    def clone(self, **kwargs) -> 'Map':
        return ConstantMap(self.t, **kwargs)


ZERO = ConstantMap(_vk.zeros(1))

ONE = ConstantMap(_vk.tensor_copy(_torch.tensor([1.0])))

POSINF = ConstantMap(_vk.tensor_copy(_torch.tensor([float('inf')])))

def as_map(value: MapLike, *, default: MapLike = ZERO) -> Map:
    """
    Converts a value to a Map.
    """
    if value is None:
        assert default is not None
        value = default
    if isinstance(value, Map):
        return value
    if isinstance(value, int) or isinstance(value, float):
        t = _vk.tensor(1)
        t[0] = value
        return ConstantMap(t)
    if isinstance(value, _vk.ViewTensor) and value.is_contiguous():
        return ConstantMap(value)
    if not isinstance(value, _torch.Tensor):
        try:
            value = _torch.as_tensor(value)
        except:
            raise TypeError(f'Type of value is not supported {type(value).__name__}')
    assert value.requires_grad == False, "Can not create a constant map from a tensor requiring grad."
    assert len(value.shape) == 1, "Can only create a constant map from a 1D tensor."
    return ConstantMap(_vk.tensor_copy(value))


# ============================
#       Composing Maps
# ============================

class PromotedMap(Map):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_promote.h",
        parameters=dict(
            map=Map,
        )
    )

    def __init__(self,
                 map: MapLike,
                 input_dim=None,
                 output_dim=None,
                 input_requires_grad=False,
                 bw_uses_output=False
                 ):
        map = as_map(map)
        if map.input_dim is not None:
            assert input_dim is None or map.input_dim == input_dim
            input_dim = map.input_dim
        assert map.output_dim is None or map.output_dim == 1
        map = map.cast(input_dim, 1, input_requires_grad, bw_uses_output)
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.map = map

    def clone(self, **kwargs) -> 'Map':
        return PromotedMap(
            self.map,
            **kwargs
        )


class SelectIndexesMap(Map):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_indices_select.h",
        parameters=dict(
            map=Map,
            indices=['OUTPUT_DIM', int]
        )
    )

    def __init__(self, map: 'MapLike', indices: _typing.List[int], input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        map = as_map(map)
        map = map.cast(
            input_dim=input_dim,
            output_dim=None,
            input_requires_grad=input_requires_grad,
            bw_uses_output=False
        )
        assert map.is_generic_input or not map.is_generic_output, "Can not select indices from a map with generic output."
        super().__init__(input_dim=map.input_dim, output_dim=len(indices), input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output, MAP_OUTPUT_DIM=map.output_dim)
        self.map = map
        for i, idx in enumerate(indices):
            self.indices[i] = idx

    def clone(self,
              **kwargs) -> 'Map':
        return SelectIndexesMap(
            self.map,
            [self.indices[i] for i in range(self.output_dim)],
            **kwargs
        )


class IdentityMap(Map, Singleton):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_identity.h"
    )

    def __init__(self, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = output_dim
        if output_dim is None:
            output_dim = input_dim
        assert input_dim == output_dim
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)

    def clone(self,
              **kwargs) -> 'Map':
        return IdentityMap(**kwargs)


class ComposeMap(Map):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_compose.h",
        parameters=dict(
            inner=Map,
            outer=Map,
        )
    )

    def __init__(self, inner: MapLike, outer: MapLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        inner = as_map(inner, default=IdentityMap.get_instance())
        outer = as_map(outer, default=IdentityMap.get_instance())
        inner = inner.cast(
            input_dim=input_dim,
            output_dim=outer.input_dim,
            input_requires_grad=input_requires_grad,
            bw_uses_output=False
        )
        inner_output_dim = inner.output_dim
        inner_requires_grad = input_requires_grad or bool(inner.deferred_info)
        outer = outer.cast(
            input_dim=inner_output_dim if inner_output_dim is not None and inner_output_dim > 1 else None,
            output_dim=output_dim,
            input_requires_grad=inner_requires_grad,
            bw_uses_output=bw_uses_output
        )
        if inner.output_dim is None and outer.input_dim is not None:
            inner = inner.cast(output_dim=outer.input_dim)
        if inner.input_dim is not None and outer.output_dim is not None:
            assert inner.output_dim is not None and outer.input_dim is not None, "Ambiguity for intermediate dimension during composing. Cast one of the two maps"
        intermediate_dim_generics = {} if inner.is_generic or outer.is_generic else {'INTERMEDIATE_DIM' : inner.output_dim}
        super().__init__(
            input_dim=inner.input_dim,
            output_dim=outer.output_dim,
            input_requires_grad=input_requires_grad,
            bw_uses_output=bw_uses_output,
            **intermediate_dim_generics,
            INNER_REQUIRES_GRAD=1 if inner_requires_grad else None
        )
        self.inner = inner
        self.outer = outer

    def clone(self,
              **kwargs) -> 'Map':
        return ComposeMap(self.inner, self.outer, **kwargs)


class DomainRangeMap(Map):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_domain_range.h",
        parameters=dict(
            range_min=_torch.Tensor,
            range_max=_torch.Tensor,
        )
    )

    def __init__(self, map: 'Map', range_min: TensorLike, range_max: TensorLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        range_min = ensure_tensor(range_min, map_dim=1)
        range_max = ensure_tensor(range_max, map_dim=1)
        map = as_map(map)
        map = map.cast(
            input_dim=input_dim,
            output_dim=output_dim,
            input_requires_grad=input_requires_grad,
            bw_uses_output=bw_uses_output
        )
        assert map.input_dim == range_min.shape[0] == range_max.shape[0]
        super().__init__(input_dim=map.input_dim, output_dim=map.output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.map = map
        self.range_min = range_min
        self.range_max = range_max

    def clone(self, **kwargs) -> 'Map':
        return DomainRangeMap(
            self.map,
            self.range_min,
            self.range_max,
            **kwargs
        )


class DomainMaskMap(Map):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_domain_mask.h",
        parameters=dict(
            map=Map,
            mask=Map,
        )
    )

    def __init__(self, map: 'MapLike', mask: 'MapLike', input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        map = as_map(map)
        mask = as_map(mask)
        if input_dim is None:
            if mask.input_dim is not None:
                input_dim = mask.input_dim
        map = map.cast(
            input_dim=input_dim,
            output_dim=output_dim,
            input_requires_grad=input_requires_grad,
            bw_uses_output=bw_uses_output
        )
        mask = mask.cast(
            input_dim=map.input_dim,
            output_dim=1,
            input_requires_grad=False,
            bw_uses_output=False
        )
        super().__init__(input_dim=map.input_dim, output_dim=map.output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.map = map
        self.mask = mask

    def clone(self,
              **kwargs) -> 'Map':
        return DomainMaskMap(
            self.map, self.mask, **kwargs
        )


# ============================
#       Map Operations
# ============================

class BinaryComponentwiseOperationMap(Map):
    __extension_info__ = None   # abstract node
    @classmethod
    def create_info(cls, path: str):
        return dict(
            path=path,
            parameters=dict(
                map_a=Map,
                map_b=Map,
            )
        )

    @staticmethod
    def _derive_signature(map_a, map_b):
        input_dim = map_a.input_dim
        if map_b.input_dim is not None:
            assert input_dim is None or map_b.input_dim == input_dim
        if map_a.output_dim == map_b.output_dim:
            output_dim = map_a.output_dim
        else:
            if map_a.output_dim is None:
                output_dim = map_b.output_dim if map_b.output_dim > 1 else None   # consider output 1 as generic still
            else:
                output_dim = map_a.output_dim if map_a.output_dim > 1 else None   # consider output 1 as generic still
        return input_dim, output_dim

    def __init__(self, map_a: MapLike, map_b: MapLike, submaps_bw_uses_output=None, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        map_a = as_map(map_a, default=ZERO)
        map_b = as_map(map_b, default=ZERO)
        map_a = map_a.cast(
            input_dim=input_dim,
            output_dim=output_dim,
            input_requires_grad=input_requires_grad,
            bw_uses_output=submaps_bw_uses_output
        )
        map_b = map_b.cast(
            input_dim=input_dim,
            output_dim=output_dim,
            input_requires_grad=input_requires_grad,
            bw_uses_output=submaps_bw_uses_output
        )
        input_dim, output_dim = BinaryComponentwiseOperationMap._derive_signature(map_a, map_b)
        map_a = map_a.cast(input_dim=input_dim, output_dim=output_dim)
        map_b = map_b.cast(input_dim=input_dim, output_dim=output_dim)
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.map_a = map_a
        self.map_b = map_b

    def clone(self, **kwargs) -> 'Map':
        return type(self)(self.map_a, self.map_b, **kwargs)


class AdditionMap(BinaryComponentwiseOperationMap):
    __extension_info__ = BinaryComponentwiseOperationMap.create_info(
        path=__INCLUDE_PATH__+"/map/map_op_add.h"
    )

    def __init__(self, map_a: MapLike, map_b: MapLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        super().__init__(map_a, map_b, False, input_dim, output_dim, input_requires_grad, bw_uses_output)


class SubtractionMap(BinaryComponentwiseOperationMap):
    __extension_info__ = BinaryComponentwiseOperationMap.create_info(
        path=__INCLUDE_PATH__+"/map/map_op_sub.h"
    )

    def __init__(self, map_a: MapLike, map_b: MapLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        super().__init__(map_a, map_b, False, input_dim, output_dim, input_requires_grad, bw_uses_output)


class MultiplicationMap(BinaryComponentwiseOperationMap):
    __extension_info__ = BinaryComponentwiseOperationMap.create_info(
        path=__INCLUDE_PATH__+"/map/map_op_mul.h"
    )

    def __init__(self, map_a: MapLike, map_b: MapLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        super().__init__(map_a, map_b, True, input_dim, output_dim, input_requires_grad, bw_uses_output)


class DivisionMap(BinaryComponentwiseOperationMap):
    __extension_info__ = BinaryComponentwiseOperationMap.create_info(
        path=__INCLUDE_PATH__+"/map/map_op_div.h"
    )

    def __init__(self, map_a: MapLike, map_b: MapLike, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        super().__init__(map_a, map_b, True, input_dim, output_dim, input_requires_grad, bw_uses_output)


class ConcatMap(Map):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_concat.h",
        parameters=dict(
            map_a = Map,
            map_b = Map
        )
    )

    def __init__(self, map_a: 'MapLike', map_b: 'MapLike', input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        map_a = as_map(map_a)
        map_b = as_map(map_b)
        if input_dim is None:
            if map_a.input_dim is not None:
                input_dim = map_a.input_dim
            elif map_b.input_dim is not None:
                input_dim = map_b.input_dim
        map_a_forced_output = None if output_dim is None or map_b.output_dim is None else output_dim - map_b.output_dim
        map_a = map_a.cast(
            input_dim=input_dim,
            output_dim=map_a_forced_output,
            input_requires_grad=input_requires_grad,
            bw_uses_output=bw_uses_output
        )
        map_b_forced_output = None if output_dim is None or map_a.output_dim is None else output_dim - map_a.output_dim
        map_b = map_b.cast(
            input_dim=input_dim,
            output_dim = map_b_forced_output,
            input_requires_grad=input_requires_grad,
            bw_uses_output=bw_uses_output
        )
        input_dim = map_a.input_dim if not map_a.is_generic_input and not map_b.is_generic_input else None
        if output_dim is None and not map_a.is_generic_output and not map_b.is_generic_output:
            output_dim = map_a.output_dim + map_b.output_dim
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            input_requires_grad=input_requires_grad,
            bw_uses_output=bw_uses_output,
            A_OUTPUT_DIM = map_a.output_dim,
            B_OUTPUT_DIM = map_b.output_dim,
        )
        self.map_a = map_a
        self.map_b = map_b

    def clone(self,
              **kwargs) -> 'Map':
        return ConcatMap(self.map_a, self.map_b, **kwargs)

# ============================
#       Sample Maps
# ============================

class Sample1DMap(Map):
    __extension_info__ = dict (
        path=__INCLUDE_PATH__+"/map/map_sample_1d.h",
        parameters=dict(
            grid=_torch.Tensor,
            align_corners=int
        )
    )

    def __init__(self,
                 grid: _typing.Union[TensorLike, deferred],
                 align_corners: bool | int = True,
                 input_dim=1, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        grid = ensure_tensor(grid, map_dim=2)
        if isinstance(grid, _torch.Tensor):
            if output_dim is None:
                output_dim = grid.shape[-1]
            assert output_dim == grid.shape[-1]
        assert input_dim == 1
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.grid = grid
        self.align_corners = 1 if align_corners else 0

    def clone(self,
              **kwargs) -> 'Map':
        return Sample1DMap(self.grid, align_corners=self.align_corners, **kwargs)


class Sample2DMap(Map):
    __extension_info__ = dict (
        path=__INCLUDE_PATH__+"/map/map_sample_2d.h",
        parameters=dict(
            grid=_torch.Tensor,
            shape=[2, int],
            align_corners=int,
        )
    )

    def __init__(self,
                 grid: _typing.Union[TensorLike, deferred],
                 align_corners: bool | int = True,
                 input_dim=2, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        grid = ensure_tensor(grid, map_dim=3)
        if isinstance(grid, _torch.Tensor):
            if output_dim is None:
                output_dim = grid.shape[-1]
            assert output_dim == grid.shape[-1]
        assert input_dim == 2
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.grid = grid
        for i in range(2):
            self.shape[i] = grid.shape[i]
        self.align_corners = int(align_corners)

    def clone(self,
              **kwargs) -> 'Map':
        return Sample2DMap(self.grid, align_corners=self.align_corners, **kwargs)


class Sample3DMap(Map):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_sample_3d.h",
        parameters=dict(
            grid=_torch.Tensor,
            shape=[3, int],
            align_corners=int,
        )
    )

    def __init__(self,
                 grid: _typing.Union[TensorLike, deferred],
                 align_corners: bool | int = True,
                 input_dim=3, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        grid = ensure_tensor(grid, map_dim=4)
        if output_dim is None:
            output_dim = grid.shape[-1]
        assert output_dim == grid.shape[-1]
        assert input_dim == 3
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.grid = grid
        for i in range(3):
            self.shape[i] = grid.shape[i]
        self.align_corners = int(align_corners)

    def clone(self,
              **kwargs) -> 'Map':
        return Sample3DMap(self.grid, self.align_corners, **kwargs)


class GPUSampleMap(Map):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_gpu_sample.h",
        parameters=dict(
            image_index=int,
            sampler_index=int,
            align_corners=int,
            shape=['INPUT_DIM', int]
        )
    )

    class Handle:
        def __init__(self, dim, id):
            self._dim = dim
            self._id = id

        @cached_property
        def id(self):
            return self._id

        def __del__(self):
            _DispatcherEngine.destroy_gpu_image(self._dim, self._id)

    def __init__(self,
                 grid: _torch.Tensor,
                 preassigned_handles = None,
                 align_corners: bool | int = True,
                 point_sampling: bool = False,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if output_dim is None:
            output_dim = grid.shape[-1]
        assert output_dim == grid.shape[-1]
        if input_dim is None:
            input_dim = len(grid.shape) - 1
        assert input_dim == len(grid.shape) - 1
        if preassigned_handles is None:
            id = _DispatcherEngine.create_gpu_image(grid.shape[:-1], output_dim)
            preassigned_handles = [GPUSampleMap.Handle(input_dim, id)]
        self.preassigned_handles = preassigned_handles
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.image_index = preassigned_handles[0].id #[h.id for h in preassigned_handles]
        self.grid = grid
        for i in range(input_dim):
            self.shape[i] = self.grid.shape[i]
        self.sampler_index = 0 if point_sampling else 1  # linear
        self.align_corners = int(align_corners)
        self.update_gpu()

    def update_gpu(self):
        _DispatcherEngine.upload_image_data(self.input_dim, self.image_index, self.grid)

    def clone(self,
              **kwargs) -> 'Map':
        return GPUSampleMap(self.grid, self.preassigned_handles, self.align_corners, self.sampler_index == 1, **kwargs)


# ============================
#       Stochastic Maps
# ============================

class UniformMap(Map, Singleton):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_uniform.h",
        stochastic=True
    )

    def clone(self, **kwargs) -> 'Map':
        return UniformMap(**kwargs)


class UniformDirectionMap(Map, Singleton):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_uniform_direction.h",
        stochastic=True
    )

    def clone(self, **kwargs) -> 'Map':
        return UniformMap(**kwargs)


class NormalMap(Map, Singleton):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_normal.h",
        stochastic=True
    )

    def clone(self, **kwargs) -> 'Map':
        return NormalMap(**kwargs)


class OctahedralProjection(Map, Singleton):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/transforms/octahedral_projection.h"
    )

    def __init__(self, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 3
        if output_dim is None:
            output_dim = 2
        assert input_dim == 3
        assert output_dim == 2
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)

    def clone(self,
              **kwargs) -> 'Map':
        return OctahedralProjection(**kwargs)


class OctahedralInverseProjection(Map, Singleton):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/transforms/octahedral_inverse_projection.h"
    )

    def __init__(self, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 2
        if output_dim is None:
            output_dim = 3
        assert input_dim == 2
        assert output_dim == 3
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)

    def clone(self,
              **kwargs) -> 'Map':
        return OctahedralInverseProjection(**kwargs)


class EquirectangularProjection(Map, Singleton):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/transforms/equirectangular_projection.h"
    )

    def __init__(self, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 3
        if output_dim is None:
            output_dim = 2
        assert input_dim == 3
        assert output_dim == 2
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)

    def clone(self,
              **kwargs) -> 'Map':
        return EquirectangularProjection(**kwargs)


class EquirectangularInverseProjection(Map, Singleton):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/transforms/equirectangular_inverse_projection.h"
    )

    def __init__(self, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 2
        if output_dim is None:
            output_dim = 3
        assert input_dim == 2
        assert output_dim == 3
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)

    def clone(self,
              **kwargs) -> 'Map':
        return EquirectangularInverseProjection(**kwargs)


class MultiplyAddTransform(Map):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/transforms/madd.h",
        parameters=dict(
            scale=_torch.Tensor,
            offset=_torch.Tensor
        )
    )

    def __init__(self, scale: _torch.Tensor, offset: _torch.Tensor,
                 input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        assert len(scale.shape) == 1
        assert len(offset.shape) == 1
        if output_dim is None:
            output_dim = scale.shape[0]
        assert output_dim == scale.shape[0] == offset.shape[0]
        if input_dim is None:
            input_dim = output_dim
        assert input_dim == output_dim
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.scale = scale
        self.offset = offset

    def clone(self,
              **kwargs) -> 'Map':
        return MultiplyAddTransform(self.scale, self.offset, **kwargs)


class SphericalHarmonics(Map):
    __extension_info__ = dict(
        path=__INCLUDE_PATH__+"/map/map_sh_eval.h",
        parameters=dict(
        )
    )

    def __init__(self, degree: int, input_dim=None, output_dim=None, input_requires_grad=False, bw_uses_output=False):
        if input_dim is None:
            input_dim = 3
        assert input_dim == 3
        num_coeffs = (degree + 1) ** 2
        if output_dim is None:
            output_dim = num_coeffs
        assert output_dim == num_coeffs
        super().__init__(input_dim=input_dim, output_dim=output_dim, input_requires_grad=input_requires_grad, bw_uses_output=bw_uses_output)
        self.degree = degree

    def clone(self,
              **kwargs) -> 'Map':
        return SphericalHarmonics(self.degree, **kwargs)


# ============================
#         Functions
# ============================


gaussian = NormalMap.get_instance()
"""
Generic map producing samples from a standard normal distribution.
"""


uniform = UniformMap.get_instance()
"""
Generic map producing samples from a uniform distribution.
"""


X = IdentityMap.get_instance()
"""
Identity map.
"""


dir2oct = OctahedralProjection.get_instance()
"""
Octahedral projection map from 3D direction to 2D.
"""

oct2dir = OctahedralInverseProjection.get_instance()
"""
Octahedral inverse projection map from 2D to 3D direction.
"""

dir2xr = EquirectangularProjection.get_instance()
"""
Equirectangular projection map from 3D direction to 2D.
"""

xr2dir = EquirectangularInverseProjection.get_instance()
"""
Equirectangular inverse projection map from 2D to 3D direction.
"""

pi = _np.pi
"""
Constant pi. Access via rdv.pi
"""

def unit2box(bmin: _vk.vec3, bmax: _vk.vec3) -> tuple[_vk.vec3, _vk.vec3]:
    """
    Returns the scale and translation to transform a cube [-1,1]^3 to the given bounding box.
    :param bmin: minimum corner of the box
    :param bmax: maximum corner of the box
    :return: scale and translation to transform [-1,1]^3 to the box defined by bmin and bmax
    """
    scale = (bmax - bmin)*0.5
    translate = (bmax + bmin)*0.5
    return scale, translate


def box2unit(bmin: _vk.vec3, bmax: _vk.vec3) -> tuple[_vk.vec3, _vk.vec3]:
    """
    Returns the scale and translation to transform a box defined by the given corners to the cube [-1,1]^3.
    :param bmin: minimum corner of the box
    :param bmax: maximum corner of the box
    :return: scale and translation to transform the box defined by bmin and bmax to [-1,1]^3
    """
    size = (bmax - bmin)
    scale = 2.0 / size
    translate = -(bmax + bmin)/size
    return scale, translate


# def unit2box(bmin: _vk.vec3, bmax: _vk.vec3) -> MultiplyAddTransform:
#     """
#     Creates a map that transforms a cube [-1,1]^3 to the given bounding box.
#     :param bmin: minimum corner of the box
#     :param bmax: maximum corner of the box
#     :return: map transforming [-1,1]^3 to the box defined by bmin and bmax
#     """
#     scale = (bmax - bmin)*0.5
#     translate = (bmax + bmin)*0.5
#     return MultiplyAddTransform(scale, translate)
#
#
# def box2unit(bmin: _vk.vec3, bmax: _vk.vec3) -> MultiplyAddTransform:
#     """
#     Creates a map that transforms a box defined by the given corners to the cube [-1,1]^3.
#     :param bmin: minimum corner of the box
#     :param bmax: maximum corner of the box
#     :return: map transforming the box defined by bmin and bmax to [-1,1]^3
#     """
#     scale = 2.0 / (bmax - bmin)
#     translate = -(bmax + bmin) * 0.5 * scale
#     return MultiplyAddTransform(scale, translate)


def grid3d_fit_box(resolution: _typing.Tuple[int, int, int], align_corners: bool = True) -> tuple[_vk.vec3, _vk.vec3]:
    """
    Returns the bmin, bmax corners to fit a 3D grid of given resolution into the unit cube [-1,1]^3.
    """
    max_dim = max(resolution)
    if align_corners:
        scale = (_vk.vec3(
            resolution[2],
            resolution[1],
            resolution[0])-1.0)/(max_dim-1)
    else:
        scale = _vk.vec3(
            resolution[2],
            resolution[1],
            resolution[0]) / max_dim
    return -scale, scale



# ===========================
#       Module Safety
# ===========================
def __setattr__(name, value):
    raise TypeError("Do not set attributes on rdv module directly.")