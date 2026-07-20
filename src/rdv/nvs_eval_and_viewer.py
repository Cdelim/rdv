# nvs_eval_and_viewer.py
#
# Two things, in one file because they're both about the REAL bicycle scene
# rather than synthetic test scenes:
#   1. A held-out-view PSNR/SSIM/LPIPS pipeline, matching the train/test
#      convention every paper you're citing uses -- this is what "compare the
#      metrics that are on the papers" actually requires.
#   2. A free-orbit interactive viewer over the real scene, for looking at
#      angles that have NO photo at all (genuinely novel, not just held-out).
#
# Needs already defined/loaded earlier in your notebook: rdv, torch, numpy,
# json, cameras_data, and your already-built method maps (gs_map, dsyg_map,
# etc.) plus the raw `positions` tensor from the .ply. This file does NOT
# rebuild scene geometry -- that's expensive and you've already done it.

# %% [markdown]
# ## STOP FIRST: check what's actually in your cameras.json
#
# Your existing cells (3, 4, 5, 11) only ever use position, rotation, width,
# height -- never a focal length or FOV. If CameraProbes has some fixed
# internal default FOV rather than reading one per camera, every render
# below will have the WRONG field of view relative to the real photos, and
# PSNR/SSIM/LPIPS will be quietly wrong without throwing any error. Run this
# first and actually look at the printed keys before trusting anything else
# in this file.

# %%
print("Keys in a camera entry:", list(cameras_data[0].keys()))
print("Example entry:", cameras_data[0])
print()
print("If you see fx/fy/FovX/FovY here: check whether rdv.CameraProbes or")
print("rdv.Sensor accepts them (try help(rdv.CameraProbes), help(rdv.Sensor)).")
print("If it doesn't accept them and your cameras have DIFFERENT fx/fy values")
print("across the list below, the metrics in this file are not trustworthy")
print("until that's resolved:")
fx_like = [k for k in cameras_data[0].keys() if 'f' in k.lower() and k.lower() not in ('id',)]
print("candidate focal-length-ish keys:", fx_like)
for k in fx_like:
    vals = [c.get(k) for c in cameras_data[:20]]
    print(f"  {k} across first 20 cameras: {vals}")

# %% [markdown]
# ## Camera pose extraction + train/test split
#
# Pose convention copied EXACTLY from your own cell 11 (forward = 3rd column
# of rotation, up = -2nd column) -- not re-derived, to avoid introducing a
# sign error your existing, working cell doesn't have.
#
# Split convention: sort by filename, hold out every 8th as test. This is
# the standard used since Mip-NeRF360 and reused by 3DGS, GOF, RaySplats,
# and StochasticSplats -- matching it is what makes your numbers comparable
# to theirs, not just internally consistent.

# %%
import os

def camera_pose_list(cam):
    pos = np.array(cam['position'], dtype=np.float32)
    rot = np.array(cam['rotation'], dtype=np.float32)
    forward = rot[:, 2]
    up = -rot[:, 1]
    target = pos + forward
    return pos, target, up


def _img_key(cam):
    for k in ('img_name', 'image_name', 'file_path', 'filename', 'image_path'):
        if k in cam:
            return cam[k]
    return str(cam.get('id', id(cam)))  # fallback: at least stable ordering


def train_test_split(cameras, test_every=8):
    ordered = sorted(cameras, key=_img_key)
    test = [c for i, c in enumerate(ordered) if i % test_every == 0]
    train = [c for i, c in enumerate(ordered) if i % test_every != 0]
    return train, test


def find_image_path(images_dir, cam, extensions=('.jpg', '.jpeg', '.png', '.JPG', '.PNG')):
    base = _img_key(cam)
    base_noext = os.path.splitext(str(base))[0]
    for ext in extensions:
        candidate = os.path.join(images_dir, base_noext + ext)
        if os.path.exists(candidate):
            return candidate
    return None


# %% [markdown]
# ## Metrics
#
# PSNR is simple enough to write by hand. SSIM and LPIPS are NOT -- both have
# fiddly-enough reference implementations (windowing, calibration weights)
# that hand-rolling them risks numbers that don't actually match what the
# papers report, which defeats the point. Use the real libraries.

# %%
import math
import numpy as np
import torch
from PIL import Image

def psnr(img, ref, max_val=1.0):
    mse = float(((img - ref) ** 2).mean())
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10((max_val ** 2) / mse)


try:
    from skimage.metrics import structural_similarity as _skimage_ssim
    def ssim(img, ref):
        return float(_skimage_ssim(img, ref, channel_axis=2, data_range=1.0))
except ImportError:
    def ssim(img, ref):
        raise RuntimeError("run: pip install scikit-image --break-system-packages -q")


_lpips_model = None
def lpips_score(img, ref, device=None):
    """img, ref: (H,W,3) numpy arrays in [0,1]. Loads the AlexNet-backed LPIPS
    net once and reuses it. First call downloads pretrained weights -- needs
    real internet access, which Colab has (this is unrelated to any sandbox
    restriction on my end; I simply can't test this specific call myself)."""
    global _lpips_model
    if _lpips_model is None:
        import lpips as _lpips_pkg
        _lpips_model = _lpips_pkg.LPIPS(net='alex')
        if device is not None:
            _lpips_model = _lpips_model.to(device)
    t_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    t_ref = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).float()
    if device is not None:
        t_img, t_ref = t_img.to(device), t_ref.to(device)
    with torch.no_grad():
        d = _lpips_model(t_img, t_ref, normalize=True)  # normalize=True: inputs are [0,1], not [-1,1]
    return float(d.item())


# %% [markdown]
# ## The evaluation loop
#
# Renders EVERY held-out test camera at ITS OWN resolution (not an arbitrary
# fixed size) with every method you pass in, loads the matching photo,
# computes all three metrics, and averages over the test set -- the same
# shape of table the papers report.

# %%
def evaluate_on_held_out(methods: dict, cameras_data, images_dir, test_every=8,
                          samples=64, device=None, max_test_views=None, verbose=True):
    """
    methods: {'GS3D': gs_map, 'DSYG': dsyg_map, ...} -- your ALREADY-BUILT maps.
    images_dir: e.g. '/content/drive/MyDrive/Gaussians/bicycle/images'
    """
    device = device or rdv.device()
    _, test_cams = train_test_split(cameras_data, test_every=test_every)
    if max_test_views is not None:
        test_cams = test_cams[:max_test_views]

    rows = []
    for cam in test_cams:
        img_path = find_image_path(images_dir, cam)
        if img_path is None:
            if verbose:
                print(f"WARNING: no photo found for camera {_img_key(cam)} in {images_dir} -- skipping")
            continue

        ref = np.asarray(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255.0

        pos, target, up = camera_pose_list(cam)
        pose_list = list(pos) + list(target) + list(up)
        camera_poses = rdv.tensor_copy(torch.tensor(pose_list, dtype=torch.float32).reshape(1, 9))
        w, h = int(cam['width']), int(cam['height'])
        sensor = rdv.Sensor(1, w, h,
            samples_location=(rdv.SampleLocation.CORNER, rdv.SampleLocation.RANDOM, rdv.SampleLocation.RANDOM),
            probes_map=rdv.CameraProbes(camera_poses=camera_poses))

        if ref.shape[0] != h or ref.shape[1] != w:
            if verbose:
                print(f"WARNING: photo is {ref.shape[1]}x{ref.shape[0]} but camera says "
                      f"{w}x{h} -- resolution mismatch, results for this view may be misleading")

        for name, model in methods.items():
            rendered = sensor.view(model, samples=samples).capture()[0]
            arr = np.clip(rendered.detach().cpu().numpy(), 0.0, 1.0)
            arr = np.flipud(arr)  # same row-order flip used elsewhere -- see interactive_overlap_viewer.py

            rows.append(dict(
                camera=_img_key(cam), method=name,
                psnr=psnr(arr, ref), ssim=ssim(arr, ref),
                lpips=lpips_score(arr, ref, device=device),
            ))

        if verbose:
            print(f"done: {_img_key(cam)}  ({len(test_cams)} test views total)")

    import pandas as pd
    df = pd.DataFrame(rows)
    summary = df.groupby('method')[['psnr', 'ssim', 'lpips']].mean()
    summary['n_views'] = df.groupby('method').size()
    return df, summary

# Run it (fill in with your actual built maps and the right image folder):
# per_view_df, summary_table = evaluate_on_held_out(
#     methods={'GS3D': gs_map, 'DSYG': dsyg_map},   # add GS3D_Ratio / GS3D_Sampled if built
#     cameras_data=cameras_data,
#     images_dir='/content/drive/MyDrive/Gaussians/bicycle/images',
#     samples=64,
# )
# print(summary_table)
# -- higher psnr/ssim is better, lower lpips is better; that's the papers' convention.


# %% [markdown]
# ## Real-scene interactive viewer -- genuinely novel (unphotographed) angles
#
# Same orbit-panel idea as interactive_overlap_viewer.py, but this one does
# NOT build any geometry -- it takes your already-built maps directly, so it
# works on the full bicycle scene without redoing an expensive BLAS/TLAS
# build. Orbit target/distance default to the scene's own extent rather than
# hardcoded toy-scene numbers, computed from `positions` so it's not a guess
# specific to one scene.

# %%
import io
try:
    import ipywidgets as widgets
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ipywidgets', '-q'])
    import ipywidgets as widgets
from IPython.display import display


def scene_orbit_defaults(positions_cpu):
    """Robust-ish centroid/radius from the raw point cloud, ignoring the
    long tail of COLMAP outlier points a straight min/max would be thrown off by."""
    centroid = positions_cpu.median(dim=0).values
    dists = (positions_cpu - centroid).norm(dim=-1)
    radius = torch.quantile(dists, 0.85).item()
    return centroid.numpy(), max(radius, 0.1)


def launch_scene_viewer(methods: dict, positions_cpu, width=420, height=420):
    """
    methods: {'GS3D': gs_map, 'DSYG': dsyg_map, ...} -- your already-built maps.
    positions_cpu: the raw (N,3) positions tensor from the .ply, on CPU,
                   used only to guess a sensible orbit target/distance.
    """
    centroid, radius = scene_orbit_defaults(positions_cpu)

    def orbit_pose(az_deg, el_deg, dist):
        az, el = np.radians(az_deg), np.radians(el_deg)
        offset = dist * np.array([np.cos(el)*np.sin(az), np.sin(el), np.cos(el)*np.cos(az)])
        return centroid + offset, centroid, np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def render_png(model, az, el, dist, samples):
        pos, tgt, up = orbit_pose(az, el, dist)
        pose_list = list(pos) + list(tgt) + list(up)
        camera_poses = rdv.tensor_copy(torch.tensor(pose_list, dtype=torch.float32).reshape(1, 9))
        sensor = rdv.Sensor(1, width, height,
            samples_location=(rdv.SampleLocation.CORNER, rdv.SampleLocation.RANDOM, rdv.SampleLocation.RANDOM),
            probes_map=rdv.CameraProbes(camera_poses=camera_poses))
        img = sensor.view(model, samples=int(samples)).capture()[0]
        arr = np.flipud(np.clip(img.detach().cpu().numpy(), 0.0, 1.0))
        pil_img = Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')
        buf = io.BytesIO(); pil_img.save(buf, format='PNG')
        return buf.getvalue()

    az_s = widgets.FloatSlider(value=0, min=0, max=360, step=2, description='azimuth', continuous_update=False)
    el_s = widgets.FloatSlider(value=10, min=-80, max=80, step=2, description='elevation', continuous_update=False)
    dist_s = widgets.FloatSlider(value=radius*2.0, min=radius*0.3, max=radius*6.0, step=radius*0.05,
                                  description='distance', continuous_update=False)
    samples_s = widgets.IntSlider(value=32, min=1, max=512, description='samples', continuous_update=False)
    method_dd = widgets.Dropdown(options=list(methods.keys()), description='method')
    img_w = widgets.Image(format='png', width=420, height=420)
    status = widgets.Label(value='')

    def _update(*_):
        status.value = 'rendering...'
        try:
            img_w.value = render_png(methods[method_dd.value], az_s.value, el_s.value, dist_s.value, samples_s.value)
            status.value = (f"az={az_s.value:.0f} el={el_s.value:.0f} dist={dist_s.value:.1f} "
                             f"(scene radius~{radius:.1f}) samples={samples_s.value}")
        except Exception as e:
            status.value = f"render failed: {e}"

    for w in (az_s, el_s, dist_s, samples_s, method_dd):
        w.observe(_update, names='value')
    display(widgets.HBox([img_w, widgets.VBox([method_dd, az_s, el_s, dist_s, samples_s, status])]))
    _update()

# Run it (positions must be the CPU tensor from your .ply-loading cell, not
# the Vulkan-copied version):
# launch_scene_viewer({'GS3D': gs_map, 'DSYG': dsyg_map}, positions)
#
# There's no photo at most of the angles you'll land on here -- that's the
# point. What you're checking for is the classic NVS failure modes: floating
# blobs, blurring, or structure that only looks right from angles near the
# training cameras. None of that shows up in the held-out-view table above,
# because every held-out view is still relatively close to some training
# view -- genuinely novel angles are a different, purely visual check.
