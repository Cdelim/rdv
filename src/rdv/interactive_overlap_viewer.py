# interactive_overlap_viewer.py
#
# Paste into new cells (split at "# %%") after everything from
# evaluation_experiments.py. Needs, already defined earlier in your notebook:
#   rdv, torch, compute_covariance, compute_inverse_covariance, rgb_to_shdc
#
# WHAT THIS ACTUALLY IS: your Colab session renders through a headless Xvfb
# virtual display -- no attached screen, no video streaming set up -- so a
# true mouse-drag, 60fps flythrough isn't something this setup supports
# without a lot of extra infrastructure. This is the honest version of
# "move around": sliders for azimuth/elevation/distance with
# continuous_update=False, so dragging fires nothing and RELEASING re-renders
# once. For a scene this small, that render is near-instant, so it feels
# responsive even though it isn't a continuous stream. A method dropdown lets
# you flip between GS3D / DSYG / GS3D_Ratio / GS3D_Sampled at the exact same
# camera angle -- that comparison is the actual point of this tool.
#
# If you want a real smooth orbit instead: export this scene to a standard
# 3DGS point_cloud.ply (trivial -- it's 2 rows) and open it in any existing
# 3DGS viewer (SIBR, or a browser-based one). You'd be navigating the
# reference rasterizer's view of the geometry, not your own shader's output,
# but the camera motion would be genuinely fluid. Say the word if you'd
# rather have that instead of or alongside this.

# %%
import io
import numpy as np
import torch
try:
    import ipywidgets as widgets
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ipywidgets', '-q'])
    import ipywidgets as widgets
from IPython.display import display
from PIL import Image


# %% [markdown]
# ## Off-axis overlap scene builder
#
# Unlike build_stack_scene (RQ1/RQ2, everything forced onto the camera axis),
# this places Gaussians at arbitrary 3D positions -- so orbiting the camera
# actually changes whether a given ray threads through both centers, one
# center, or neither, the way real overlap ambiguity shows up in practice.

# %%
def build_overlap_scene(positions_3d, opacities, colors_rgb, scale=0.6, device=None):
    device = device or rdv.device()
    N = len(opacities)
    positions = torch.as_tensor(positions_3d, dtype=torch.float32).reshape(N, 3)
    opacities_t = torch.as_tensor(opacities, dtype=torch.float32)
    colors_rgb_t = torch.as_tensor(colors_rgb, dtype=torch.float32).reshape(N, 3)

    scales = torch.full((N, 3), float(scale))
    rotations = torch.zeros(N, 4)
    rotations[:, 0] = 1.0  # identity quaternion

    covs = compute_covariance(scales, rotations)
    inv_covs = compute_inverse_covariance(scales, rotations)
    sh_dc = rgb_to_shdc(colors_rgb_t)
    f_rest = torch.zeros(N, 45)

    def _vk3(t):
        return rdv.tensor_copy(rdv.vec3(t).to(device))

    def _vkflat(t):
        return rdv.tensor_copy(t.to(device))

    return dict(
        positions_vk=_vk3(positions), colors_vk=_vk3(sh_dc), scales_vk=_vk3(scales),
        f_rest_vk=_vkflat(f_rest), inv_covs_vk=_vkflat(inv_covs),
        opacities_vk=_vkflat(opacities_t), covs_vk=_vkflat(covs),
    )


def make_map(map_cls, tensors):
    m = map_cls(
        tensors['positions_vk'], tensors['colors_vk'],
        inv_covs=tensors['inv_covs_vk'], opacities=tensors['opacities_vk'],
        scales=tensors['scales_vk'], f_rest=tensors['f_rest_vk'], covs=tensors['covs_vk'],
    )
    m.build_ads()
    return m


# %% [markdown]
# ## Orbit camera

# %%
def orbit_camera_pose(azimuth_deg, elevation_deg, distance, target=(0.0, 0.0, 0.0)):
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    offset = distance * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    pos = np.array(target, dtype=np.float32) + offset
    return pos, np.array(target, dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)


def make_sensor_for_pose(pos, target, up, width, height):
    pose_list = list(pos) + list(target) + list(up)
    camera_poses = rdv.tensor_copy(torch.tensor(pose_list, dtype=torch.float32).reshape(1, 9))
    return rdv.Sensor(
        1, width, height,
        samples_location=(rdv.SampleLocation.CORNER, rdv.SampleLocation.RANDOM, rdv.SampleLocation.RANDOM),
        probes_map=rdv.CameraProbes(camera_poses=camera_poses),
    )


def render_view_png(model, azimuth, elevation, distance, samples, width=220, height=220, target=(0.0, 0.0, 0.0)):
    pos, tgt, up = orbit_camera_pose(azimuth, elevation, distance, target=target)
    sensor = make_sensor_for_pose(pos, tgt, up, width, height)
    img = sensor.view(model, samples=int(samples)).capture()[0]
    arr = np.clip(img.detach().cpu().numpy(), 0.0, 1.0)
    arr = np.flipud(arr)  # matches the plt.gca().invert_yaxis() convention used
                           # elsewhere in your notebook -- row 0 here is the
                           # bottom of the image, PIL/PNG expect row 0 = top
    pil_img = Image.fromarray((arr * 255).astype(np.uint8), mode='RGB')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return buf.getvalue()


# %% [markdown]
# ## Default scene: two overlapping Gaussians, off the camera axis
#
# Positioned symmetrically about the origin so the natural look-at target
# (0,0,0) sits exactly in their overlap region at almost every orbit angle --
# the one place peak-based hit selection can disagree with a properly
# sampled or numerically-marched answer. Increase `scale` or decrease the
# 1.0 separation below for heavier overlap; the opposite for less.

# %%
overlap_scene = build_overlap_scene(
    positions_3d=[[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
    opacities=[0.7, 0.7],
    colors_rgb=[[0.9, 0.15, 0.15], [0.15, 0.25, 0.9]],  # A red, B blue -- matches the diagram
    scale=0.6,
)

# Build every available method's map ONCE here (acceleration structure only
# depends on scene geometry, never on the camera) -- the interactive panel
# below only ever rebuilds the sensor, not the scene.
_method_classes = {
    'GS3D (delta, stochastic)': 'GS3D',
    'DSYG (deterministic, sorted)': 'DSYG',
    'GS3D_Ratio (deterministic, weighted)': 'GS3D_Ratio',
    'GS3D_Sampled (properly-sampled)': 'GS3D_Sampled',
}
available_methods = {}
missing = []
for label, clsname in _method_classes.items():
    if hasattr(rdv, clsname):
        available_methods[label] = make_map(getattr(rdv, clsname), overlap_scene)
    else:
        missing.append(clsname)

assert available_methods, "None of GS3D/DSYG/GS3D_Ratio/GS3D_Sampled are available in rdv."
if missing:
    print(f"Note: {', '.join(missing)} not found -- add their files to your rdv "
          f"project (see earlier headers) to compare against them too. "
          f"Continuing with: {list(available_methods.keys())}")


# %% [markdown]
# ## The panel

# %%
az_slider = widgets.FloatSlider(value=25, min=0, max=360, step=2, description='azimuth', continuous_update=False)
el_slider = widgets.FloatSlider(value=15, min=-85, max=85, step=2, description='elevation', continuous_update=False)
dist_slider = widgets.FloatSlider(value=4.0, min=1.5, max=12.0, step=0.25, description='distance', continuous_update=False)
samples_slider = widgets.IntSlider(value=64, min=1, max=1024, step=1, description='samples',
                                    continuous_update=False)
method_dropdown = widgets.Dropdown(options=list(available_methods.keys()), description='method')

image_widget = widgets.Image(format='png', width=340, height=340)
status_label = widgets.Label(value='')
note_label = widgets.Label(value='samples only visibly matters for the stochastic methods (GS3D, GS3D_Sampled)')

def _update(*_):
    status_label.value = 'rendering...'
    try:
        model = available_methods[method_dropdown.value]
        png_bytes = render_view_png(model, az_slider.value, el_slider.value,
                                     dist_slider.value, samples_slider.value)
        image_widget.value = png_bytes
        status_label.value = (f"az={az_slider.value:.0f}  el={el_slider.value:.0f}  "
                               f"dist={dist_slider.value:.1f}  samples={samples_slider.value}")
    except Exception as e:
        status_label.value = f"render failed: {e}"

for w in (az_slider, el_slider, dist_slider, samples_slider, method_dropdown):
    w.observe(_update, names='value')

controls = widgets.VBox([method_dropdown, az_slider, el_slider, dist_slider, samples_slider,
                          status_label, note_label])
display(widgets.HBox([image_widget, controls]))
_update()

# Try this comparison: set method to DSYG, find an angle where the red/blue
# overlap band is clearly visible, note the color balance in that band, then
# switch to GS3D_Sampled at the same angle without touching the sliders.
# The shift you see there is the same ~10-18% relative bias measured in
# overlap_ground_truth.py, now something you can actually look at.
