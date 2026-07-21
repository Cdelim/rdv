# interactive_realtime_viewer.py
#
# Real-time, actually-interactive flythrough of a reconstructed GS3D / DSYG /
# GS3D_Ratio / GS3D_Sampled scene -- LOCAL ONLY. This needs a real GPU with
# Vulkan and a real attached display; it will NOT run on Colab (no display
# to stream to). For Colab, use the slider-based viewer in
# notebook_sandbox/nvs_evaluation.ipynb instead -- that one is the honest
# equivalent for a headless session.
#
# Every frame: read the current camera pose, build a fresh rdv.Sensor
# pointed at it (same "rebuild sensor per view" pattern used everywhere else
# in this project -- there is no Sensor.update()), render, display, and read
# input for the next frame. The acceleration structure (build_ads()) is
# built once at startup and never rebuilt -- only the camera pose tensor
# changes per frame, and rdv's compute-pipeline cache is keyed by shape/
# generics rather than by tensor values, so the same pipeline is reused
# across frames automatically.
#
# GS3D and GS3D_Sampled are STOCHASTIC -- at the low samples-per-pixel
# needed for real-time framerates they will look visibly noisy. That's
# expected, not a bug; raise samples with ']' or switch to a deterministic
# method (DSYG / GS3D_Ratio) to see a clean image at the same angle.
#
# Uses OpenCV (`pip install opencv-python`) for the window, image display,
# and keyboard/mouse input -- NOT vulky's own windowing. vulky is a
# separate, uninstalled dependency in this environment and this project has
# no existing usage of its window API to follow, so guessing at its
# signature would be exactly the kind of unfounded API assumption to avoid.
# OpenCV's highgui window/mouse-callback API is well-documented and used
# here instead.
#
# Known limitation, stated plainly rather than glossed over: cv2.waitKey()
# reports one key EVENT per call, not a "currently held" boolean the way a
# game engine's input system would. Movement below rides on your OS/window
# manager's own key-repeat while a key is held, which is good enough for
# exploring a scene but not perfectly smooth. If your terminal's repeat
# delay is long, tap repeatedly instead of holding for finer control.
#
# Usage:
#   pip install opencv-python plyfile
#   python interactive_realtime_viewer.py \
#       --ply /path/to/point_cloud.ply \
#       --cameras /path/to/cameras.json      # optional, only used for the starting pose
#
# Controls:
#   W / A / S / D   move forward / left / back / right
#   E / Q           move up / down
#   drag left mouse button   look around
#   1 / 2 / 3 / 4   switch method: GS3D / DSYG / GS3D_Ratio / GS3D_Sampled
#                   (only the ones present in your rdv build are offered)
#   [ / ]           decrease / increase samples per pixel
#   - / =           decrease / increase movement speed
#   P               print the current camera pose (pos/yaw/pitch) to the console
#   Esc             quit

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

import rdv


WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
WINDOW_NAME = "rdv interactive viewer  (Esc to quit, P to print pose)"


# --------------------------------------------------------------------------
# Scene loading -- deliberately duplicated from notebook_sandbox/
# nvs_evaluation.ipynb rather than imported, matching this project's existing
# convention of self-contained standalone scripts (see the header comments
# in evaluation_experiments.py / nvs_eval_and_viewer.py for the same choice).
# --------------------------------------------------------------------------

def compute_inverse_covariance(scales, rotations):
    inv_sq_scales = 1.0 / (scales ** 2 + 1e-7)

    r = rotations[:, 0]
    x = rotations[:, 1]
    y = rotations[:, 2]
    z = rotations[:, 3]

    R = torch.zeros((rotations.shape[0], 3, 3), device=scales.device)
    R[:, 0, 0] = 1.0 - 2.0 * (y**2 + z**2)
    R[:, 0, 1] = 2.0 * (x * y - r * z)
    R[:, 0, 2] = 2.0 * (x * z + r * y)
    R[:, 1, 0] = 2.0 * (x * y + r * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x**2 + z**2)
    R[:, 1, 2] = 2.0 * (y * z - r * x)
    R[:, 2, 0] = 2.0 * (x * z - r * y)
    R[:, 2, 1] = 2.0 * (y * z + r * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x**2 + y**2)

    inv_cov = torch.zeros((scales.shape[0], 6), device=scales.device)
    for i in range(3):
        for j in range(i, 3):
            val = (R[:, i, 0] * inv_sq_scales[:, 0] * R[:, j, 0] +
                   R[:, i, 1] * inv_sq_scales[:, 1] * R[:, j, 1] +
                   R[:, i, 2] * inv_sq_scales[:, 2] * R[:, j, 2])
            idx = i * 3 - (i * (i + 1)) // 2 + j
            inv_cov[:, idx] = val
    return inv_cov


def compute_covariance(scales, rotations):
    sq_scales = scales ** 2

    r = rotations[:, 0]
    x = rotations[:, 1]
    y = rotations[:, 2]
    z = rotations[:, 3]

    R = torch.zeros((rotations.shape[0], 3, 3), device=scales.device)
    R[:, 0, 0] = 1.0 - 2.0 * (y**2 + z**2)
    R[:, 0, 1] = 2.0 * (x * y - r * z)
    R[:, 0, 2] = 2.0 * (x * z + r * y)
    R[:, 1, 0] = 2.0 * (x * y + r * z)
    R[:, 1, 1] = 1.0 - 2.0 * (x**2 + z**2)
    R[:, 1, 2] = 2.0 * (y * z - r * x)
    R[:, 2, 0] = 2.0 * (x * z - r * y)
    R[:, 2, 1] = 2.0 * (y * z + r * x)
    R[:, 2, 2] = 1.0 - 2.0 * (x**2 + y**2)

    cov = torch.zeros((scales.shape[0], 6), device=scales.device)
    for i in range(3):
        for j in range(i, 3):
            val = (R[:, i, 0] * sq_scales[:, 0] * R[:, j, 0] +
                   R[:, i, 1] * sq_scales[:, 1] * R[:, j, 1] +
                   R[:, i, 2] * sq_scales[:, 2] * R[:, j, 2])
            idx = i * 3 - (i * (i + 1)) // 2 + j
            cov[:, idx] = val
    return cov


def load_scene(ply_path):
    from plyfile import PlyData

    print(f"Loading {ply_path} ...")
    plydata = PlyData.read(ply_path)
    v = plydata['vertex']

    x = torch.tensor(v['x'].copy(), dtype=torch.float32)
    y = torch.tensor(v['y'].copy(), dtype=torch.float32)
    z = torch.tensor(v['z'].copy(), dtype=torch.float32)
    positions = torch.stack((x, y, z), dim=-1)

    f_dc_0 = torch.tensor(v['f_dc_0'].copy(), dtype=torch.float32)
    f_dc_1 = torch.tensor(v['f_dc_1'].copy(), dtype=torch.float32)
    f_dc_2 = torch.tensor(v['f_dc_2'].copy(), dtype=torch.float32)
    sh_dc = torch.stack((f_dc_0, f_dc_1, f_dc_2), dim=-1)

    rest_names = [f'f_rest_{i}' for i in range(45)]
    f_rest = torch.stack([torch.tensor(v[name].copy(), dtype=torch.float32) for name in rest_names], dim=-1)

    scale_names = ['scale_0', 'scale_1', 'scale_2']
    scales = torch.stack([torch.tensor(v[name].copy(), dtype=torch.float32) for name in scale_names], dim=-1)
    scales = torch.exp(scales).clamp(min=1e-6, max=5.0)

    rot_names = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
    rotations = torch.stack([torch.tensor(v[name].copy(), dtype=torch.float32) for name in rot_names], dim=-1)
    rotations = torch.nn.functional.normalize(rotations, dim=-1)

    opacities = torch.sigmoid(torch.tensor(v['opacity'].copy(), dtype=torch.float32))

    mask = opacities.squeeze() > 0.02

    inv_covs = compute_inverse_covariance(scales, rotations)
    covs = compute_covariance(scales, rotations)

    positions = positions[mask]
    sh_dc = sh_dc[mask]
    f_rest = f_rest[mask]
    scales = scales[mask]
    opacities = opacities[mask]
    inv_covs = inv_covs[mask]
    covs = covs[mask]

    print(f"Loaded {positions.shape[0]} Gaussians after culling ({mask.numel() - mask.sum().item()} removed).")

    positions_vk = rdv.tensor_copy(rdv.vec3(positions).to(rdv.device()))
    colors_vk = rdv.tensor_copy(rdv.vec3(sh_dc).to(rdv.device()))
    scales_vk = rdv.tensor_copy(rdv.vec3(scales).to(rdv.device()))
    f_rest_vk = rdv.tensor_copy(f_rest.to(rdv.device()))
    inv_covs_vk = rdv.tensor_copy(inv_covs.to(rdv.device()))
    opacities_vk = rdv.tensor_copy(opacities.to(rdv.device()))
    covs_vk = rdv.tensor_copy(covs.to(rdv.device()))

    def build(map_cls):
        m = map_cls(
            positions_vk, colors_vk,
            inv_covs=inv_covs_vk, opacities=opacities_vk,
            scales=scales_vk, f_rest=f_rest_vk, covs=covs_vk,
        )
        m.build_ads()
        return m

    methods = {'GS3D': build(rdv.GS3D), 'DSYG': build(rdv.DSYG)}
    if hasattr(rdv, 'GS3D_Ratio'):
        methods['GS3D_Ratio'] = build(rdv.GS3D_Ratio)
    if hasattr(rdv, 'GS3D_Sampled'):
        methods['GS3D_Sampled'] = build(rdv.GS3D_Sampled)

    return methods, positions


def scene_orbit_defaults(positions_cpu):
    centroid = positions_cpu.median(dim=0).values
    dists = (positions_cpu - centroid).norm(dim=-1)
    radius = torch.quantile(dists, 0.85).item()
    return centroid.numpy().astype(np.float64), max(radius, 0.1)


# --------------------------------------------------------------------------
# Free-fly camera
# --------------------------------------------------------------------------

class CameraState:
    def __init__(self, pos, yaw, pitch):
        self.pos = np.array(pos, dtype=np.float64)
        self.yaw = float(yaw)
        self.pitch = float(pitch)

    def basis(self):
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        forward = np.array([cp * sy, sp, cp * cy], dtype=np.float64)
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, WORLD_UP)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        return forward, right, up


def yaw_pitch_from_forward(forward):
    forward = forward / np.linalg.norm(forward)
    pitch = math.asin(float(np.clip(forward[1], -1.0, 1.0)))
    yaw = math.atan2(float(forward[0]), float(forward[2]))
    return yaw, pitch


def initial_camera_state(cameras_path, positions_cpu):
    if cameras_path is not None and os.path.exists(cameras_path):
        import json
        with open(cameras_path, 'r') as f:
            cameras_data = json.load(f)
        cam = cameras_data[0]
        pos = np.array(cam['position'], dtype=np.float64)
        rot = np.array(cam['rotation'], dtype=np.float64)
        forward = rot[:, 2]  # same convention as camera_pose_list() elsewhere in this project
        yaw, pitch = yaw_pitch_from_forward(forward)
        print(f"Starting from cameras.json[0]: pos={pos}")
        return CameraState(pos, yaw, pitch)

    centroid, radius = scene_orbit_defaults(positions_cpu)
    pos = centroid + np.array([0.0, 0.0, -radius * 2.0], dtype=np.float64)
    forward = centroid - pos
    yaw, pitch = yaw_pitch_from_forward(forward)
    print(f"No --cameras given: starting {radius * 2.0:.2f} units back from the scene centroid {centroid}")
    return CameraState(pos, yaw, pitch)


def make_sensor(width, height, camera_poses):
    """
    IMPORTANT: rdv.Sensor's shape arguments (after the leading camera-count)
    are in (rows, cols) = (height, width) order, matching numpy/OpenCV's
    image convention -- NOT (width, height). Passing them as (width, height)
    silently transposes the render: a landscape scene comes out portrait,
    width and height swapped.

    Traced this from perspective_camera_sensors.h + capture_forward.h:
    `_input[0]` (the shader's horizontal/sx ray deflection) is generated
    from the FASTEST-varying / LAST axis of the output tensor, and that
    axis's size is controlled by Sensor's THIRD positional argument, not
    its second. aspect_ratio is set explicitly too -- CameraProbes defaults
    to 1.0 (square), which would otherwise stretch any non-square render
    since perspective_camera_sensors.h scales only the horizontal (sx) ray
    deflection by aspect_ratio, leaving the vertical one unscaled.
    """
    return rdv.Sensor(
        1, height, width,
        samples_location=(rdv.SampleLocation.CORNER, rdv.SampleLocation.RANDOM, rdv.SampleLocation.RANDOM),
        probes_map=rdv.CameraProbes(camera_poses=camera_poses, aspect_ratio=width / height),
    )


def render_frame(model, state, samples, width, height):
    forward, right, up = state.basis()
    target = state.pos + forward
    pose_list = list(state.pos) + list(target) + list(up)
    camera_poses = rdv.tensor_copy(torch.tensor(pose_list, dtype=torch.float32).reshape(1, 9))
    sensor = make_sensor(width, height, camera_poses)
    img = sensor.view(model, samples=int(samples)).capture()[0]
    arr = np.clip(img.detach().cpu().numpy(), 0.0, 1.0)
    # BOTH axes need flipping, not just one -- found this out empirically while
    # debugging the companion notebooks against real ground truth photos:
    # (1) vertical: rdv renders with row 0 = bottom of the scene, an established
    #     convention elsewhere in this project.
    # (2) horizontal: the camera's "right" direction, from cross(up, forward) in
    #     perspective_camera_sensors.h, comes out mirrored relative to how the
    #     point cloud is laid out -- most likely a left/right-handedness mismatch
    #     between the scene's coordinate convention and what the shader's
    #     cross-product assumes. This is a property of the scene + shader
    #     pairing (not of this file's own yaw/pitch camera math), so it applies
    #     here too even though this viewer builds its pose differently than the
    #     notebooks' cameras.json-derived one.
    # ascontiguousarray avoids a crash when this later goes into cv2/torch,
    # neither of which accept negative-stride arrays.
    return np.ascontiguousarray(arr[::-1, ::-1, :])


class MouseLook:
    def __init__(self):
        self.dragging = False
        self.last_x = 0
        self.last_y = 0
        self.delta_x = 0.0
        self.delta_y = 0.0

    def callback(self, event, x, y, flags, param):
        import cv2
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_x, self.last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.delta_x += (x - self.last_x)
            self.delta_y += (y - self.last_y)
            self.last_x, self.last_y = x, y

    def consume(self):
        dx, dy = self.delta_x, self.delta_y
        self.delta_x = 0.0
        self.delta_y = 0.0
        return dx, dy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ply', required=True, help='Path to point_cloud.ply')
    parser.add_argument('--cameras', default=None, help='Path to cameras.json (optional, only sets the starting pose)')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--samples', type=int, default=4, help='Samples per pixel (raise with "]" at runtime for a cleaner look)')
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("This viewer needs OpenCV: pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    methods, positions_cpu = load_scene(args.ply)
    method_names = [n for n in ('GS3D', 'DSYG', 'GS3D_Ratio', 'GS3D_Sampled') if n in methods]
    print(f"Methods available (1-{len(method_names)}): {method_names}")

    state = initial_camera_state(args.cameras, positions_cpu)
    _, scene_radius = scene_orbit_defaults(positions_cpu)

    active_idx = 0
    samples = args.samples
    speed = max(scene_radius * 0.5, 0.5)  # scene-units per second
    mouse_sensitivity = 0.004  # radians per pixel dragged
    pitch_limit = math.radians(89.0)

    mouse = MouseLook()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse.callback)

    print("Controls: WASD move, E/Q up/down, drag LMB to look, 1-4 switch method, "
          "[ / ] samples, - / = speed, P print pose, Esc quit.")

    last_time = time.perf_counter()
    while True:
        now = time.perf_counter()
        dt = max(now - last_time, 1e-4)
        last_time = now

        dx, dy = mouse.consume()
        state.yaw -= dx * mouse_sensitivity
        state.pitch = float(np.clip(state.pitch - dy * mouse_sensitivity, -pitch_limit, pitch_limit))

        key = cv2.waitKey(1) & 0xFF
        forward, right, up = state.basis()
        if key in (ord('w'), ord('W')):
            state.pos += forward * speed * dt
        elif key in (ord('s'), ord('S')):
            state.pos -= forward * speed * dt
        elif key in (ord('a'), ord('A')):
            state.pos -= right * speed * dt
        elif key in (ord('d'), ord('D')):
            state.pos += right * speed * dt
        elif key in (ord('e'), ord('E')):
            state.pos += WORLD_UP * speed * dt
        elif key in (ord('q'), ord('Q')):
            state.pos -= WORLD_UP * speed * dt
        elif key == ord('['):
            samples = max(1, samples - 1)
        elif key == ord(']'):
            samples += 1
        elif key == ord('-'):
            speed = max(0.01, speed * 0.8)
        elif key in (ord('='), ord('+')):
            speed *= 1.25
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            idx = key - ord('1')
            if idx < len(method_names):
                active_idx = idx
        elif key in (ord('p'), ord('P')):
            print(f"pos={state.pos.tolist()}  yaw={state.yaw:.4f}  pitch={state.pitch:.4f}")
        elif key == 27:  # Esc
            break

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        model_name = method_names[active_idx]
        frame = render_frame(methods[model_name], state, samples, args.width, args.height)
        frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        fps = 1.0 / dt
        hud = f"{model_name}  spp={samples}  speed={speed:.2f}  {fps:.0f} fps"
        cv2.putText(frame_bgr, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, frame_bgr)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
