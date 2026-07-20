# evaluation_experiments.py
#
# Paste these cells (split at the "# %%" markers) into new cells at the end of
# your existing notebook, in order. They assume the following already ran
# earlier in the session (they do, in your current notebook):
#   - `import rdv`, `import torch`
#   - `compute_covariance` and `compute_inverse_covariance` are defined
#
# NOTE: the `sensor` object your notebook already built from cameras.json is
# pointed at the real bicycle scene and will NOT see the synthetic scenes
# used below. Use `build_stack_sensor()` (defined further down) for every
# RQ1/RQ2/RQ3 call instead -- do not reuse the bicycle `sensor` here.
#
# Everything here is written as functions with their own local variable names
# (stack_*, gs_map, dsyg_map, ratio_map, ...) so it will NOT collide with or
# overwrite your real bicycle-scene tensors (positions, scales, opacities, ...).
#
# RQ1 and RQ2 need nothing beyond what you already have (rdv.GS3D, rdv.DSYG).
# RQ3 needs decomposition_tracking_GS_ratio.h and _gaussian_splats_ratio.py
# added to your rdv project first (see the header comment in that .py file).

# %% [markdown]
# ## Helpers: SH color convention, synthetic stack scenes, analytic ground truth

# %%
import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# Degree-0 real spherical harmonic basis constant. The shaders compute
#   gaussian_color = colors[i] * sh_coefs[0]  (+ higher-order f_rest terms)
#   final_rgb      = clamp(gaussian_color + 0.5, 0, 1)
# so a target RGB has to be converted into this "SH-DC" space before being
# handed to GS3D / DSYG / GS3D_Ratio.
SH_C0 = 0.28209479177387814

def rgb_to_shdc(rgb):
    rgb = torch.as_tensor(rgb, dtype=torch.float32)
    return (rgb - 0.5) / SH_C0


def build_stack_scene(opacities, colors_rgb, spacing=0.15, scale=2.5, device=None):
    """
    Builds N isotropic Gaussians centered exactly ON the camera axis (x=y=0),
    stacked front-to-back along +Z, starting at z=0. Because every Gaussian's
    center sits exactly on the ray, the ray-Gaussian "power" term is exactly
    zero for all of them, which makes each one's rendered alpha exactly equal
    to its assigned opacity -- this is what makes an exact, hand-computable
    ground truth possible (see analytic_front_to_back_composite below).

    opacities  : length-N sequence, per-Gaussian opacity in (0, 1)
    colors_rgb : (N, 3) sequence, target RGB in [0, 1] (only the DC/base SH
                 term is used -- f_rest is zeroed, so color is view-independent)
    scale      : isotropic Gaussian scale (world units). Keep this comfortably
                 larger than the angular footprint, at this distance, of the
                 central image patch you intend to analyze -- otherwise pixels
                 near the patch edges won't see the same on-axis geometry as
                 the center, and the "many pixels = many independent trials"
                 trick used below breaks down. If your convergence plot looks
                 noisy in a way that doesn't shrink with samples, increase
                 `scale` or move the camera farther away first.

    Returns a dict of Vulkan-ready tensors plus the plain (CPU) opacities/
    colors in front-to-back order, needed for the analytic ground truth.
    """
    device = device or rdv.device()
    N = len(opacities)
    opacities_t = torch.as_tensor(opacities, dtype=torch.float32)
    colors_rgb_t = torch.as_tensor(colors_rgb, dtype=torch.float32).reshape(N, 3)

    positions = torch.zeros(N, 3)
    positions[:, 2] = torch.arange(N, dtype=torch.float32) * spacing

    scales = torch.full((N, 3), float(scale))
    rotations = torch.zeros(N, 4)
    rotations[:, 0] = 1.0  # identity quaternion (w=1, x=y=z=0): axis-aligned, isotropic

    covs = compute_covariance(scales, rotations)
    inv_covs = compute_inverse_covariance(scales, rotations)
    sh_dc = rgb_to_shdc(colors_rgb_t)
    f_rest = torch.zeros(N, 45)

    def _vk3(t):
        return rdv.tensor_copy(rdv.vec3(t).to(device))

    def _vkflat(t):
        return rdv.tensor_copy(t.to(device))

    return dict(
        positions_vk=_vk3(positions),
        colors_vk=_vk3(sh_dc),
        scales_vk=_vk3(scales),
        f_rest_vk=_vkflat(f_rest),
        inv_covs_vk=_vkflat(inv_covs),
        opacities_vk=_vkflat(opacities_t),
        covs_vk=_vkflat(covs),
        opacities_cpu=opacities_t,
        colors_cpu=colors_rgb_t,
    )


def make_map(map_cls, tensors):
    m = map_cls(
        tensors['positions_vk'], tensors['colors_vk'],
        inv_covs=tensors['inv_covs_vk'], opacities=tensors['opacities_vk'],
        scales=tensors['scales_vk'], f_rest=tensors['f_rest_vk'], covs=tensors['covs_vk'],
    )
    m.build_ads()
    return m


def analytic_front_to_back_composite(opacities, colors_rgb):
    """
    Exact expected pixel color for the on-axis stack, front (index 0) to back,
    against a black background. This is real ground truth -- derived by hand
    from the alpha-compositing recursion, not by cross-checking against DSYG.
    """
    opacities = torch.as_tensor(opacities, dtype=torch.float32)
    colors_rgb = torch.as_tensor(colors_rgb, dtype=torch.float32)
    T = 1.0
    color = torch.zeros(3)
    for a, c in zip(opacities, colors_rgb):
        color = color + T * a * c
        T = T * (1.0 - a)
    return color


def time_capture(sensor, model, samples, n_warmup=1, n_timed=3):
    """Warm up once (compilation / first-launch overhead, as in your existing
    notebook cells), then average `n_timed` timed captures."""
    img = None
    for _ in range(n_warmup):
        img = sensor.view(model, samples=samples).capture()[0]
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        img = sensor.view(model, samples=samples).capture()[0]
        times.append(time.perf_counter() - t0)
    return img, sum(times) / len(times)


def central_patch(img, patch=8):
    H, W, _ = img.shape
    cy, cx = H // 2, W // 2
    h = patch // 2
    return img[cy - h: cy + h, cx - h: cx + h, :]


def build_stack_sensor(width=64, height=64, camera_z=-6.0):
    """
    IMPORTANT: the `sensor` object already in your notebook is built from
    cameras.json and points at the real bicycle scene -- it will NOT see
    these synthetic on-axis stacks, which sit at the origin at a completely
    different scale. Use THIS sensor (a direct port of your own commented-out
    cell 10) for every RQ1/RQ2/RQ3 call below, not the bicycle one.

    Looks straight down +Z from (0,0,camera_z) at the origin, matching
    build_stack_scene's placement of Gaussian 0 at z=0. `camera_z` should be
    comfortably in front of the whole stack (i.e. more negative than
    -N*spacing for your largest N).
    """
    pos = [0.0, 0.0, camera_z]
    target = [0.0, 0.0, 0.0]
    up = [0.0, 1.0, 0.0]
    pose_list = pos + target + up
    camera_poses = rdv.tensor_copy(torch.tensor(pose_list, dtype=torch.float32).reshape(1, 9))
    return rdv.Sensor(
        1, width, height,
        samples_location=(rdv.SampleLocation.CORNER, rdv.SampleLocation.RANDOM, rdv.SampleLocation.RANDOM),
        probes_map=rdv.CameraProbes(camera_poses=camera_poses),
    )

# Build this ONCE and pass it to every rq*_ function below:
# stack_sensor = build_stack_sensor()
# If N grows large enough that camera_z=-6 no longer sees the whole stack,
# call build_stack_sensor(camera_z=...) again with a more negative value.


# %% [markdown]
# ## RQ1 -- Correctness: does GS3D converge to the analytic ground truth?
#
# Uses only `rdv.GS3D` (already implemented) and a 4-Gaussian on-axis stack,
# a direct extension of the toy scene you already had commented out.

# %%
def rq1_convergence_test(sensor, opacities_gt=(0.5, 0.2, 1.0, 0.6),
                          colors_gt=((0.9, 0.1, 0.1), (0.1, 0.8, 0.1),
                                     (0.1, 0.1, 0.9), (0.9, 0.9, 0.1)),
                          sample_counts=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                          patch=8, spacing=0.15, scale=2.5):
    tensors = build_stack_scene(opacities_gt, colors_gt, spacing=spacing, scale=scale)
    gs_map = make_map(rdv.GS3D, tensors)
    gt = analytic_front_to_back_composite(opacities_gt, colors_gt)

    rmse_per_s = []
    for s in sample_counts:
        img = sensor.view(gs_map, samples=s).capture()[0]
        block = central_patch(img, patch).cpu()
        rmse = ((block - gt.view(1, 1, 3)) ** 2).mean().sqrt().item()
        rmse_per_s.append(rmse)
        print(f"samples={s:5d}  RMSE vs. analytic ground truth = {rmse:.4e}")

    return list(sample_counts), rmse_per_s, gt


def plot_rq1(sample_counts, rmse_per_s):
    s = np.asarray(sample_counts, dtype=float)
    rmse = np.asarray(rmse_per_s, dtype=float)
    plt.figure()
    plt.loglog(s, rmse, 'o-', label='GS3D (delta-tracking-style)')
    ref = rmse[0] * np.sqrt(s[0] / s)
    plt.loglog(s, ref, '--', color='gray', label='ideal 1/sqrt(N) reference (slope -1/2)')
    plt.xlabel('samples per pixel')
    plt.ylabel('RMSE vs. analytic ground truth')
    plt.title('RQ1: convergence of the decomposition-tracking estimator')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.show()

# Run it (stack_sensor, NOT the bicycle sensor -- see build_stack_sensor above):
# stack_sensor = build_stack_sensor()
# sample_counts, rmse_per_s, gt = rq1_convergence_test(stack_sensor)
# plot_rq1(sample_counts, rmse_per_s)
#
# What to look for: RMSE should fall roughly along the dashed reference line.
# If it flattens out well above zero at high sample counts, something in the
# estimator is biased -- fix that before trusting anything downstream.
# If it fits the slope but sits noticeably ABOVE the dashed line at low sample
# counts, that's fine; the reference line's height is anchored to your first
# data point and is only there to check the *slope*, not the absolute offset.


# %% [markdown]
# ## RQ2 -- Does decomposition tracking actually help as density/opacity grows?
#
# Compares GS3D (stochastic) against DSYG (deterministic, sorted) on stacks of
# increasing size N and opacity. No new shaders needed -- this is testable today.

# %%
def rq2_density_sweep(sensor, N_values=(1, 2, 4, 8, 16, 32, 64),
                       opacity_values=(0.2, 0.5, 0.9, 0.99),
                       gs_samples=32, patch=8, spacing=0.15, scale=2.5):
    results = []
    for N in N_values:
        for op in opacity_values:
            opacities = [op] * N
            colors = [[0.9, 0.2, 0.2]] * N  # color is arbitrary here; only N and
                                             # opacity are the variables under test
            tensors = build_stack_scene(opacities, colors, spacing=spacing, scale=scale)

            dsyg_map = make_map(rdv.DSYG, tensors)
            gt_img, dsyg_time = time_capture(sensor, dsyg_map, samples=1)

            gs_map = make_map(rdv.GS3D, tensors)
            gs_img, gs_time = time_capture(sensor, gs_map, samples=gs_samples)

            gt_block = central_patch(gt_img, patch).cpu()
            gs_block = central_patch(gs_img, patch).cpu()

            noise = gs_block.std(dim=(0, 1)).mean().item()
            bias = (gs_block.mean(dim=(0, 1)) - gt_block.mean(dim=(0, 1))).abs().mean().item()

            results.append(dict(N=N, opacity=op, dsyg_time=dsyg_time, gs_time=gs_time,
                                 gs_noise=noise, gs_bias=bias))
            print(f"N={N:3d} op={op:.2f}  DSYG={dsyg_time*1e3:6.2f}ms  "
                  f"GS3D({gs_samples}spp)={gs_time*1e3:6.2f}ms  "
                  f"noise={noise:.4f}  bias={bias:.4f}")
    return results


def plot_rq2(results):
    import pandas as pd
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for op, g in df.groupby('opacity'):
        axes[0].plot(g['N'], g['gs_noise'], 'o-', label=f'opacity={op}')
    axes[0].set_xlabel('overlapping Gaussians (N)')
    axes[0].set_ylabel('GS3D per-pixel std, central patch')
    axes[0].set_title('Noise vs. local density')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    for op, g in df.groupby('opacity'):
        axes[1].plot(g['N'], g['dsyg_time'] * 1e3, '--', label=f'DSYG, op={op}')
        axes[1].plot(g['N'], g['gs_time'] * 1e3, '-', label=f'GS3D, op={op}')
    axes[1].set_xlabel('overlapping Gaussians (N)')
    axes[1].set_ylabel('render time (ms)')
    axes[1].set_title('Cost vs. local density (dashed = DSYG, solid = GS3D)')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    return df

# Run it (reuse the same stack_sensor from RQ1; for large N in N_values,
# you may need build_stack_sensor(camera_z=...) with a more negative value
# so the whole stack stays in view):
# rq2_results = rq2_density_sweep(stack_sensor)
# rq2_df = plot_rq2(rq2_results)
#
# Watch particularly for: (a) does gs_time grow more slowly than dsyg_time as
# N grows -- that's the "sorting cost" argument; (b) does gs_noise at FIXED
# samples grow with opacity/N -- that's the "reduced noise in near-opaque
# media" claim, stated as its converse (more noise where DSYG-equivalent
# density is high). If N approaches 256, also check whether DSYG's hit buffer
# is truncating (see the MAX_HITS note in DontSplashYourGaussians.h) before
# treating its output as ground truth at that density.


# %% [markdown]
# ## RQ3 -- Delta-tracking-style vs. ratio-tracking-style
#
# Requires decomposition_tracking_GS_ratio.h and _gaussian_splats_ratio.py to
# be added to your rdv project (see the header of that .py file for exactly
# where). Once `rdv.GS3D_Ratio` exists, this reuses the RQ2 stacks.

# %%
def rq3_delta_vs_ratio(sensor, N_values=(1, 2, 4, 8, 16, 32, 64),
                        opacity_values=(0.5, 0.9, 0.99),
                        delta_sample_grid=(4, 16, 64, 256),
                        patch=8, spacing=0.15, scale=2.5):
    assert hasattr(rdv, 'GS3D_Ratio'), (
        "rdv.GS3D_Ratio not found -- add decomposition_tracking_GS_ratio.h and "
        "_gaussian_splats_ratio.py to your rdv project first (see that file's header)."
    )

    results = []
    for N in N_values:
        for op in opacity_values:
            opacities = [op] * N
            colors = [[0.9, 0.2, 0.2]] * N
            tensors = build_stack_scene(opacities, colors, spacing=spacing, scale=scale)

            ratio_map = make_map(rdv.GS3D_Ratio, tensors)
            # samples=1 is enough for GS3D_Ratio: its blending step is
            # deterministic, the only thing "samples" affects is pixel-jitter
            # anti-aliasing from the sensor, if any.
            ratio_img, ratio_time = time_capture(sensor, ratio_map, samples=1)
            ratio_block = central_patch(ratio_img, patch).cpu()

            row = dict(N=N, opacity=op, ratio_time=ratio_time)
            for s in delta_sample_grid:
                delta_map = make_map(rdv.GS3D, tensors)
                delta_img, delta_time = time_capture(sensor, delta_map, samples=s)
                delta_block = central_patch(delta_img, patch).cpu()
                mse_vs_ratio = ((delta_block - ratio_block) ** 2).mean().item()
                row[f'delta_time_s{s}'] = delta_time
                row[f'delta_mse_vs_ratio_s{s}'] = mse_vs_ratio

            results.append(row)
            summary = "  ".join(f"s={s}: mse={row[f'delta_mse_vs_ratio_s{s}']:.2e} "
                                 f"({row[f'delta_time_s{s}']*1e3:.1f}ms)" for s in delta_sample_grid)
            print(f"N={N:3d} op={op:.2f}  ratio={ratio_time*1e3:6.2f}ms  |  delta: {summary}")
    return results

# Run it (after adding the new files to your rdv project and restarting the
# Colab runtime so the new shader gets compiled in):
# rq3_results = rq3_delta_vs_ratio(stack_sensor)
#
# Sanity check FIRST: at op=0.5, N=1 (a single, clearly semi-transparent
# Gaussian, no overlap), delta at high samples and ratio should already agree
# closely -- if they don't, something is wrong with one of the two shaders,
# and it's worth finding out which before reading anything into the N>1 trend.
#
# The actual RQ3 question: as N and opacity grow, does delta need increasingly
# more samples/time to reach the same mse_vs_ratio level that a small N
# needed? That growth curve is the professor's hypothesis, made concrete.


# %% [markdown]
# ## Logging results to disk (so plots/tables can be regenerated without
# ## re-rendering everything)

# %%
def save_results_csv(results, path):
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    return df

# Example:
# save_results_csv(rq2_results, '/content/drive/MyDrive/Gaussians/rq2_results.csv')
# save_results_csv(rq3_results, '/content/drive/MyDrive/Gaussians/rq3_results.csv')


# %% [markdown]
# ## RQ4 starters -- broader positioning (sketches, not finished experiments)

# %%
def psnr(img, ref, max_val=1.0):
    mse = ((img - ref) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10((max_val ** 2) / mse)

# For real image-quality numbers you need the ORIGINAL TRAINING PHOTOGRAPHS,
# not just cameras.json. Standard 3DGS output includes an `images/` folder
# next to point_cloud.ply -- check:
#   /content/drive/MyDrive/Gaussians/bicycle/images/
# If it's there, load the photo matching cameras_data[0] (same index you used
# to build `sensor`) and compare:
#   from PIL import Image
#   ref_img = torch.from_numpy(np.array(Image.open(images_path)) / 255.0).float()
#   print("PSNR vs. photograph:", psnr(image.cpu(), ref_img))
# For SSIM/LPIPS: `pip install scikit-image` gives skimage.metrics.structural_similarity;
# `pip install lpips` gives a learned perceptual metric if you want one.

# Secondary-ray demo sketch: GS3D's FORWARD map takes an arbitrary (position,
# direction) pair, not specifically a camera ray -- so a shadow ray costs
# nothing new to try. You need a real surface hit point `hit_pos` (e.g. from
# a first camera render, by unprojecting a pixel using its depth) and a light
# position `light_pos`:
#
#   shadow_origin = hit_pos + 1e-3 * normalize(light_pos - hit_pos)  # avoid self-hit
#   shadow_dir = normalize(light_pos - hit_pos)
#   shadow_ray = torch.cat([shadow_origin, shadow_dir]).view(1, 6).to(rdv.device())
#   visibility = gs_map(shadow_ray)   # near-zero transmittance along the ray => shadowed
#
# This is the cheapest possible demonstration that ray tracing bought you
# something rasterization structurally can't do -- worth one figure in
# Chapter 6 even without a full relighting pipeline.
