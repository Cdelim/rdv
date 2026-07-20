# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`rdv` ("rendervous") is a differentiable volume/Gaussian-splat rendering framework used for master's
thesis research. Python classes describe GPU compute kernels (GLSL-ish shader code in `.h` files under
`src/rdv/include/`); the framework JIT-generates Vulkan compute shaders from them at runtime and wires
them into PyTorch's autograd. There is no build step in the traditional sense — kernels are compiled
lazily the first time a given `Map`/`Compute` signature is dispatched.

The GPU/Vulkan backend is provided by an external package, **`vulky`** (imported everywhere as `_vk`),
which is not part of this repo and is not installed in every environment. Most of the runtime code here
cannot actually execute (create a device, dispatch a kernel) without `vulky` installed and a Vulkan-capable
GPU (CUDA is used for the torch tensor side). Treat missing `vulky` / missing CUDA as expected in a plain
checkout — read/edit the code, don't assume you can run it.

## Commands

There is no packaging metadata (no `pyproject.toml`/`setup.py`), no lint config, and no CI. There isn't an
established "run the tests" command captured anywhere in the repo — the tests under `tests/` are
`unittest`-based but hard-require a CUDA device and a working `vulky`/Vulkan setup (e.g.
`tests/test_maps.py` allocates `device='cuda'` tensors directly), so they are effectively local-machine-only
integration tests, not something to run in a sandboxed environment. If asked to run them, use
`python -m unittest discover tests` and expect failure without a GPU + `vulky`.

`sandbox/*.py` are ad hoc manual scripts (not pytest/unittest) for interactively poking at `vulky`/`rdv`
primitives — read them as usage examples, don't expect them to run headlessly either.

## Architecture

### The `Map` abstraction (`src/rdv/_core.py`)

A `Map` is a differentiable function R^n → R^m. Concrete maps subclass `_core.Map` (metaclass `_MapMeta`,
built on `_ComputeMeta`) and declare `__extension_info__`, e.g.:

```python
class SomeMap(_core.Map):
    __extension_info__ = dict(
        path=_core.__INCLUDE_PATH__ + '/some/kernel.h',  # or code="... GLSL-ish source ..."
        parameters=dict(a=_torch.Tensor, b=float, ...),   # becomes a GPU-side struct
        generics=dict(...),                                # compile-time #defines
        stochastic=True,                                   # kernel uses random()
    )
```

Each unique combination of map type + generics + parameter compute-ids gets a `rdv_signature`; the
`_DispatcherEngine` (also in `_core.py`) compiles a kernel per signature once, generating GLSL by
concatenating `compute_template.h` + per-map kernel code (built from the `.h` source referenced above,
wrapped with `system/push_signatures.h` / `pop_signatures.h`) and dispatching it through `vulky`.

Maps compose like ordinary functions/operators: `+ - * / |` (concat), `.then()`, `.after()`, `.cast(...)`,
`.domain_range()`, `.domain_mask()`, `.promote()`. Calling a map (`map(input, **deferred_tensors)`) routes
through `_MapEvalFunction`, a `torch.autograd.Function` that dispatches `_MapForwardEvalCompute` /
`_MapBackwardEvalCompute` — this is how GPU kernel evaluation gets forward/backward autograd support.

**Deferred parameters** (`deferred(key, shape)`): a way to bind a *named* tensor into a map's parameters at
call time rather than construction time, so the same compiled kernel can be reused across different actual
tensors (and so gradients can flow into arbitrary bound tensors via `_DeferredParametersManager`).

### `Compute` (also `_core.py`)

Lower-level building block than `Map`: a `Compute` subclass (`bind()` + `result()`) represents a one-off
GPU dispatch over `N` threads that isn't itself a differentiable R^n→R^m map (e.g. filling a buffer,
capturing a sensor). `Map` evaluation and `Sensor.capture()` are themselves implemented on top of `Compute`
subclasses (`_MapForwardEvalCompute`, `_CaptureForwardCompute`, etc).

### Shader source layout (`src/rdv/include/`)

- `core.h` — GPU-side prelude: pointer macros, math constants/helpers, RNG.
- `compute_template.h` — wraps a compute kernel body with the local-size layout, system uniform buffer,
  and bindless sampler/texture arrays; `#include`d by every generated shader.
- `system/` — plumbing for pushing/popping per-map signatures and forward/backward eval entry points.
- `map/` — generic `Map` kernels (sampling, arithmetic ops, compose, domain range/mask, SH eval, etc.) —
  these back the Python classes exported near the top of `_core.py` (`ComposeMap`, `Sample2DMap`, ...).
- `rendering/`, `volume_rendering/`, `transforms/`, `signatures/`, `traits/` — the actual
  rendering/volume-integration kernels (camera sensors, phase functions, transmittance/free-flight
  estimators, BVH-based ray-marching, Gaussian-splat decomposition tracking, etc.) backing the Python
  classes in `_rendering.py`, `_volume_rendering.py`, `_gaussian_splats*.py`, `_dont_splash_your_gaussians.py`.

Python classes and their `.h` kernels are tightly coupled 1:1 — when editing a `Map`'s behavior you
generally need to change both the Python `parameters=dict(...)` declaration/`__init__` and the
corresponding `.h` file (parameter names/order and types must match).

### Research-specific modules

- `_gaussian_splats.py` (`GS3D`), `_dont_splash_your_gaussians.py` (`DSYG`),
  `_gaussian_splats_ratio.py` (`GS3D_Ratio`), `_gaussian_splats_sampled.py` (`GS3D_Sampled`) — different
  volumetric-integration strategies over the same 3D Gaussian-splat scene representation
  (positions/colors/covariances/opacities/spherical-harmonics `f_rest`), each backed by its own
  `include/volume_rendering/*.h` kernel. These are the core objects of the thesis's comparison.
- `evaluation_experiments.py`, `nvs_eval_and_viewer.py`, `overlap_ground_truth.py`,
  `interactive_overlap_viewer.py` — **not standalone scripts**. They are written as notebook-cell blocks
  (split on `# %%` markers) meant to be pasted into `src/notebook_sandbox/3DGSDecompostionTrck.ipynb`,
  and assume state already built earlier in that notebook session (loaded scene tensors, camera data,
  helper functions, etc). Read the header comments in each file before touching it — they document what
  notebook state they depend on and known gotchas (e.g. `nvs_eval_and_viewer.py`'s header warns that
  camera FOV handling needs verifying against `cameras.json` before trusting any metric it produces).
- `src/notebook_sandbox/*.ipynb` — the actual interactive research notebooks; primary entry point for
  running experiments end-to-end.

### Output/data directories

`output/`, `Gaussians/` hold generated images and Gaussian-splat scene assets (gitignored/untracked bulk
data) — not source, just experiment artifacts.
