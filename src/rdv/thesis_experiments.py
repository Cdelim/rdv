# thesis_experiments.py
#
# ONE script, run once, produces every table and figure for the thesis.
#
# HOW TO USE AT THE LAB
#   1. Set the CONFIG block below.
#   2. Build your three maps in the notebook as usual, then:
#
#         from thesis_experiments import run_all
#         run_all(maps={
#             'GS3D_Sampled': sampled_map,
#             'DSYG':         dsyg_map,
#             'Sun':          sun_map,
#         }, sensor=sensor)
#
#   3. Everything lands in RESULTS_DIR: .csv for the numbers, .tex for
#      paste-ready tables, .png for figures.
#
# Running it with no arguments (`run_all()`) executes only the analytic
# experiments, which need nothing but numpy and produce the core accuracy
# results. Good for checking the script works before touching the GPU.

import os
import csv
import math
import time
import numpy as np

# ============================ CONFIG ============================

RESULTS_DIR = "./thesis_results"

# Held-out view evaluation. Leave IMAGES_DIR as None to skip it.
IMAGES_DIR   = None      # e.g. "/path/to/room/images_2"
CAMERAS_JSON = None      # e.g. "/path/to/room/cameras.json"
TEST_EVERY   = 8         # standard 3DGS/Mip-NeRF360 held-out convention
MAX_TEST_VIEWS = 8       # cap, keeps runtime sane; None for all

# Sample counts for the stochastic convergence study
SAMPLE_COUNTS  = (1, 4, 16, 64, 256, 1024)
REFERENCE_SPP  = 4096    # self-reference for GS3D_Sampled convergence

# Which methods are stochastic (need many samples) vs deterministic (1 is exact)
STOCHASTIC = {'GS3D_Sampled', 'Sun'}

# Crop for the fine-detail figure, (y0, y1, x0, x1). Set after looking at one
# render; None uses the whole frame.
DETAIL_CROP = None

MC_TRIALS = 300_000      # trials for the analytic Monte Carlo experiments

# ================================================================


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------------- utilities ---------------------------

def giles_erfinv(x):
    """Giles (GPU Computing Gems 2010). Validated to 2.4e-7 vs scipy."""
    x = np.asarray(x, dtype=np.float64)
    w = -np.log((1.0 - x) * (1.0 + x))
    out = np.empty_like(x)
    m = w < 5.0
    w1 = w[m] - 2.5
    p1 = np.full_like(w1, 2.81022636e-08)
    for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
              -0.00125372503, -0.00417768164, 0.246640727, 1.50140941):
        p1 = c + p1 * w1
    out[m] = p1 * x[m]
    w2 = np.sqrt(w[~m]) - 3.0
    p2 = np.full_like(w2, -0.000200214257)
    for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
              -0.0076224613, 0.00943887047, 1.00167406, 2.83297682):
        p2 = c + p2 * w2
    out[~m] = p2 * x[~m]
    return out


def probit(u):
    u = np.clip(u, 1e-9, 1 - 1e-9)
    return math.sqrt(2.0) * float(giles_erfinv(np.array([2.0 * u - 1.0]))[0])


def ncdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def save_table(name, header, rows, caption, label, align=None):
    """Writes both a .csv and a paste-ready booktabs .tex."""
    _ensure_dir()
    with open(os.path.join(RESULTS_DIR, name + ".csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(header)
        wtr.writerows(rows)

    align = align or ("l" + "r" * (len(header) - 1))
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{" + caption + "}", r"\label{tab:" + label + "}",
             r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(str(h) for h in header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(RESULTS_DIR, name + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  [{name}]  {caption}")
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(header)]
    print("  " + "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(header)))
    print("  " + "  ".join("-" * widths[i] for i in range(len(header))))
    for r in rows:
        print("  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))


# ==================================================================
#  PART A -- analytic experiments. No GPU, no rdv, numpy only.
# ==================================================================

def _mk(A, tstar, alpha, color):
    tau = -math.log(1.0 - alpha)
    return dict(A=A, tstar=tstar, alpha=alpha, tau=tau,
                sigma_peak=tau * math.sqrt(A / (2.0 * math.pi)),
                color=np.asarray(color, dtype=np.float64))


def _scene(separation=0.6, alpha=0.6):
    return [_mk(8.0, 1.0, alpha, [1, 0, 0]),
            _mk(5.0, 1.0 + separation, alpha, [0, 0, 1])]


def true_ground_truth(gs, t0=-6.0, t1=10.0, n=1_500_000):
    """Fine-step marching of the JOINT density. The reference everything
    else is measured against."""
    t = np.linspace(t0, t1, n); dt = t[1] - t[0]
    st = np.zeros(n); sc = np.zeros((n, 3))
    for g in gs:
        s = g['sigma_peak'] * np.exp(-0.5 * g['A'] * (t - g['tstar']) ** 2)
        st += s; sc += s[:, None] * g['color'][None, :]
    T = np.exp(-np.cumsum(st) * dt)
    Tp = np.concatenate(([1.0], T[:-1]))
    return np.sum(Tp[:, None] * sc * dt, axis=0)


def peak_only(gs):
    """What DSYG and Sun et al. both converge to: each primitive credited at
    its own peak, composited in depth order."""
    T, col = 1.0, np.zeros(3)
    for g in sorted(gs, key=lambda g: g['tstar']):
        col = col + T * g['alpha'] * g['color']
        T *= (1.0 - g['alpha'])
    return col


def mc_estimator(gs, mode, n=MC_TRIALS, seed=0):
    """mode: 'sun'        -- accept at the peak, closest accepted wins
             'density'    -- accept, then place via probit(u)   (two draws)
             'freeflight' -- one closed-form draw does both     (one draw)"""
    rng = np.random.default_rng(seed)
    acc = np.zeros(3); draws = 0
    for _ in range(n):
        best_t, best_c = np.inf, None
        for g in gs:
            if mode == 'sun':
                draws += 1
                if rng.random() < g['alpha'] and g['tstar'] < best_t:
                    best_t, best_c = g['tstar'], g['color']
            elif mode == 'density':
                draws += 1
                if rng.random() <= g['alpha']:
                    draws += 1
                    t = g['tstar'] + probit(rng.random()) / math.sqrt(g['A'])
                    if 0 < t < best_t:
                        best_t, best_c = t, g['color']
            else:
                draws += 1
                u = rng.random()
                Phi = -math.log(1.0 - u) / g['tau']
                if Phi < 1.0:
                    t = g['tstar'] + probit(Phi) / math.sqrt(g['A'])
                    if 0 < t < best_t:
                        best_t, best_c = t, g['color']
        if best_c is not None:
            acc += best_c
    return acc / n, draws / n


def _relerr(v, gt):
    return float((np.abs(v - gt) / np.maximum(gt, 1e-9)).max() * 100.0)


def A1_overlap_accuracy():
    """Headline accuracy result: all three methods vs true volumetric GT."""
    gs = _scene()
    gt = true_ground_truth(gs)
    rows = [["True volumetric integral", np.round(gt, 4).tolist(), "--", "reference"]]
    for mode, label in [('sun', "Sun et al. / DSYG (peak-only)"),
                        ('density', "GS3D\\_Sampled (density profile)"),
                        ('freeflight', "GS3D\\_Sampled (free-flight)")]:
        v, _ = mc_estimator(gs, mode)
        rows.append([label, np.round(v, 4).tolist(), f"{_relerr(v, gt):.2f}",
                     "stochastic"])
    save_table("A1_overlap_accuracy",
               ["Method", "RGB", "Max rel. err (\\%)", "Kind"], rows,
               "Accuracy against the true volumetric integral on a two-Gaussian "
               "overlap scene with a hand-derived reference.", "overlap_accuracy")
    return rows


def A2_separation_sweep():
    """The bias is an OVERLAP effect: it vanishes as primitives separate."""
    rows = []
    for sep in (0.3, 0.6, 1.0, 1.5, 2.0, 3.0):
        gs = _scene(separation=sep)
        gt = true_ground_truth(gs)
        pk = peak_only(gs)
        ff, _ = mc_estimator(gs, 'freeflight', n=MC_TRIALS // 2)
        rows.append([f"{sep:.1f}", f"{_relerr(pk, gt):.2f}", f"{_relerr(ff, gt):.2f}"])
    save_table("A2_separation_sweep",
               ["Separation", "Peak-only err (\\%)", "GS3D\\_Sampled err (\\%)"],
               rows,
               "Peak-only compositing error as a function of primitive overlap. "
               "The error is specific to the overlap regime and vanishes once "
               "primitives separate.", "separation_sweep")
    return rows


def A3_opacity_sweep():
    """Free-flight vs density-profile sampling across the density regimes."""
    rows = []
    for a in (0.3, 0.6, 0.9, 0.99):
        gs = _scene(alpha=a)
        gt = true_ground_truth(gs)
        pk = peak_only(gs)
        d, dd = mc_estimator(gs, 'density', n=MC_TRIALS // 2)
        f, df = mc_estimator(gs, 'freeflight', n=MC_TRIALS // 2)
        rows.append([f"{a:.2f}", f"{_relerr(pk, gt):.2f}",
                     f"{_relerr(d, gt):.2f}", f"{_relerr(f, gt):.2f}",
                     f"{dd:.2f}", f"{df:.2f}"])
    save_table("A3_opacity_sweep",
               ["Opacity", "Peak-only (\\%)", "Density profile (\\%)",
                "Free-flight (\\%)", "Draws/ray (dens.)", "Draws/ray (f.f.)"],
               rows,
               "Estimator error and random-number cost across density regimes. "
               "The closed-form free-flight sample uses one draw per primitive "
               "regardless of opacity, and is markedly more accurate in the "
               "near-opaque regime.", "opacity_sweep")
    return rows


def A4_depth_bias():
    """Free-flight sampling pulls hits toward the camera; quantify it."""
    rng = np.random.default_rng(0)
    rows = []
    for op in (0.1, 0.3, 0.6, 0.9, 0.99):
        tau = -math.log(1.0 - op)
        u = rng.random(200_000)
        Phi = -np.log(1.0 - u) / tau
        hit = Phi < 1.0
        z = np.array([probit(p) for p in Phi[hit][:20000]])
        rows.append([f"{op:.2f}", f"{tau:.3f}", f"{np.median(z):.3f}",
                     f"{z.mean():.3f}"])
    save_table("A4_depth_bias",
               ["Opacity", "$\\tau$", "Median shift ($\\sigma$)",
                "Mean shift ($\\sigma$)"], rows,
               "Displacement of the sampled interaction point relative to the "
               "peak, in units of the primitive's extent along the ray. Negative "
               "values lie toward the camera: the near half of a dense primitive "
               "shadows the far half.", "depth_bias")
    return rows


def A5_numerical_stability():
    """The power cancellation finding, and the stable residual form."""
    rng = np.random.default_rng(0)
    f32 = np.float32

    def R_of(q):
        q = q / np.linalg.norm(q); w_, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-w_*z), 2*(x*z+w_*y)],
                         [2*(x*y+w_*z), 1-2*(x*x+z*z), 2*(y*z-w_*x)],
                         [2*(x*z-w_*y), 2*(y*z+w_*x), 1-2*(x*x+y*y)]])

    e_alg, e_stab = [], []
    cull_alg = cull_stab = total = 0
    for _ in range(60_000):
        sp = np.exp(rng.uniform(np.log(0.05), np.log(0.6)))
        st = sp / np.exp(rng.uniform(np.log(10), np.log(200)))
        R = R_of(rng.normal(size=4))
        M = R @ np.diag([1/sp**2, 1/sp**2, 1/st**2]) @ R.T
        dist = rng.uniform(2.0, 40.0)
        ang = np.radians(rng.uniform(0.5, 20.0))
        w = np.array([np.cos(ang), -np.sin(ang), 0.0]); w /= np.linalg.norm(w)
        centre = w * dist + R[:, 0] * rng.normal() * sp * 0.5
        d = -centre
        A64 = float(w @ M @ w)
        if A64 <= 1e-6:
            continue
        B64 = float(w @ M @ d)
        dp64 = d + (-B64 / A64) * w
        p64 = -0.5 * float(dp64 @ M @ dp64)

        Mf, wf, df = M.astype(f32), w.astype(f32), d.astype(f32)
        A = f32(wf @ Mf @ wf)
        if A <= f32(1e-6):
            continue
        B = f32(wf @ Mf @ df); C = f32(df @ Mf @ df)
        total += 1
        p_alg = f32(-0.5) * f32(C - f32(B * B) / A)
        dpf = (df + f32(-B / A) * wf).astype(f32)
        p_stab = f32(-0.5) * f32(dpf @ Mf @ dpf)
        e_alg.append(abs(p64 - float(p_alg)))
        e_stab.append(abs(p64 - float(p_stab)))
        if p64 > -4.0 and float(p_alg) < -4.0:
            cull_alg += 1
        if p64 > -4.0 and float(p_stab) < -4.0:
            cull_stab += 1

    e_alg = np.array(e_alg); e_stab = np.array(e_stab)
    rows = [
        ["Algebraic $C - B^2/A$", f"{np.median(e_alg):.2e}",
         f"{np.percentile(e_alg,99):.2e}", f"{e_alg.max():.2e}",
         f"{100*cull_alg/total:.2f}"],
        ["Residual $d'^{T} M d'$", f"{np.median(e_stab):.2e}",
         f"{np.percentile(e_stab,99):.2e}", f"{e_stab.max():.2e}",
         f"{100*cull_stab/total:.2f}"],
    ]
    save_table("A5_numerical_stability",
               ["Formulation", "Median err", "p99 err", "Max err",
                "Spurious discards (\\%)"], rows,
               "Single-precision error in the squared Mahalanobis distance for "
               "grazing rays through anisotropic primitives, against a "
               "double-precision reference. A spurious discard is a primitive "
               "wrongly rejected by the support cutoff.", "numerical_stability")
    return rows


# ==================================================================
#  PART B -- GPU experiments. Need the built maps and a sensor.
# ==================================================================

def _render_timed(sensor, model, samples):
    sensor.view(model, samples=samples).capture()[0]      # warm-up
    t0 = time.perf_counter()
    img = sensor.view(model, samples=samples).capture()[0]
    return img.detach(), time.perf_counter() - t0


def _img_psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return (float('inf') if mse <= 0 else 10.0 * math.log10(1.0 / mse)), mse


def G1_render_time(maps, sensor, spp_stochastic=64):
    rows = []
    for name, model in maps.items():
        n = spp_stochastic if name in STOCHASTIC else 1
        _, t = _render_timed(sensor, model, n)
        rows.append([name.replace('_', '\\_'), n, f"{t:.3f}",
                     f"{t/n*1000:.2f}"])
    save_table("G1_render_time",
               ["Method", "spp", "Time (s)", "ms per sample"], rows,
               "Render time on the evaluation scene. Deterministic methods are "
               "exact at one sample per pixel; stochastic methods are timed at "
               "the sample count used for the quality comparison.", "render_time")
    return rows


def G2_convergence(maps, sensor, crop=None):
    """Noise vs sample count for the stochastic methods, self-referenced so
    that modelling differences between methods do not contaminate it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir()
    rows = []
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in [n for n in maps if n in STOCHASTIC]:
        model = maps[name]
        ref, _ = _render_timed(sensor, model, REFERENCE_SPP)
        if crop:
            ref = ref[crop[0]:crop[1], crop[2]:crop[3]]
        xs, ys = [], []
        for n in SAMPLE_COUNTS:
            img, t = _render_timed(sensor, model, n)
            if crop:
                img = img[crop[0]:crop[1], crop[2]:crop[3]]
            p, _ = _img_psnr(img, ref)
            rows.append([name.replace('_', '\\_'), n, f"{p:.2f}", f"{t:.3f}"])
            xs.append(n); ys.append(p)
        ax.semilogx(xs, ys, 'o-', label=name)
    ax.set_xlabel("samples per pixel")
    ax.set_ylabel(f"PSNR vs. {REFERENCE_SPP}-spp reference (dB)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "G2_convergence.png"), dpi=160)
    plt.close(fig)

    save_table("G2_convergence",
               ["Method", "spp", "PSNR vs. reference (dB)", "Time (s)"], rows,
               "Convergence of the stochastic estimators. Each method is "
               "compared against its own high-sample-count render, so the "
               "measurement isolates sampling noise from modelling differences.",
               "convergence")
    return rows


def _load_gt_image(images_dir, img_name, target_hw):
    """Loads the photograph for a camera and resizes it to the render's size."""
    from PIL import Image
    stem = os.path.splitext(str(img_name))[0]
    match = [f for f in os.listdir(images_dir)
             if os.path.splitext(f)[0] == stem]
    if not match:
        return None
    im = Image.open(os.path.join(images_dir, match[0])).convert("RGB")
    if im.size != (target_hw[1], target_hw[0]):
        im = im.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def G3_qualitative(maps, sensor, crop=None, gt=None):
    """Paper-style comparison figure: one column per method, plus ground truth
    when available. Top row is the full frame, bottom row the detail crop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir()
    panels = {}
    if gt is not None:
        panels["Ground truth"] = np.clip(gt, 0, 1)
    for name, model in maps.items():
        n = 256 if name in STOCHASTIC else 1
        img, _ = _render_timed(sensor, model, n)
        arr = np.clip(img.cpu().numpy(), 0, 1)
        label = f"{name} ({n} spp)" if name in STOCHASTIC else f"{name} (1 spp)"
        panels[label] = arr
        plt.imsave(os.path.join(RESULTS_DIR, f"G3_{name}.png"), arr)

    nrows = 2 if crop else 1
    fig, axes = plt.subplots(nrows, len(panels),
                             figsize=(4.2 * len(panels), 3.6 * nrows),
                             squeeze=False)
    for j, (label, arr) in enumerate(panels.items()):
        axes[0][j].imshow(arr); axes[0][j].set_title(label, fontsize=11)
        axes[0][j].axis('off')
        if crop:
            y0, y1, x0, x1 = crop
            axes[0][j].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                               fill=False, ec='red', lw=1.2))
            axes[1][j].imshow(arr[y0:y1, x0:x1])
            axes[1][j].axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "G3_comparison.png"), dpi=200)
    plt.close(fig)
    print(f"\n  [G3] wrote G3_comparison.png and per-method renders to "
          f"{RESULTS_DIR}")
    return panels


def _metrics(pred, gt):
    """PSNR, SSIM, LPIPS -- the three reported by 3DGS, Sun et al. and the
    Mip-NeRF360 line of work. SSIM/LPIPS degrade to None if the packages
    are missing rather than aborting the run."""
    mse = float(((pred - gt) ** 2).mean())
    psnr = float('inf') if mse <= 0 else 10.0 * math.log10(1.0 / mse)

    ssim = None
    try:
        from skimage.metrics import structural_similarity
        ssim = float(structural_similarity(gt, pred, channel_axis=2,
                                           data_range=1.0))
    except Exception:
        pass

    lp = None
    try:
        import torch, lpips
        if not hasattr(_metrics, "_net"):
            _metrics._net = lpips.LPIPS(net='alex')
        def to_t(a):
            t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float()
            return t * 2.0 - 1.0
        with torch.no_grad():
            lp = float(_metrics._net(to_t(pred), to_t(gt)).item())
    except Exception:
        pass
    return psnr, ssim, lp


def G4_held_out(maps, make_sensor_fn=None, images_dir=None, cameras_json=None):
    """PSNR / SSIM / LPIPS against held-out photographs, averaged over the
    test split (every TEST_EVERY-th view, the standard 3DGS convention).

    make_sensor_fn(cam_dict) -> sensor. Pass your notebook's own make_sensor,
    since it is the thing that knows how to read the per-camera focal length.

    CAVEAT: these numbers are only as good as that focal-length handling. If
    the sensor's field of view does not match the camera the photograph came
    from, every value here is biased and nothing else will reveal it."""
    import json
    images_dir = images_dir or IMAGES_DIR
    cameras_json = cameras_json or CAMERAS_JSON
    if not images_dir or not cameras_json:
        print("\n  [G4] skipped -- IMAGES_DIR / CAMERAS_JSON not set.")
        return None
    if make_sensor_fn is None:
        print("\n  [G4] skipped -- pass make_sensor_fn=make_sensor so each "
              "test camera can be rendered at its own pose and FOV.")
        return None

    with open(cameras_json) as f:
        cams = json.load(f)
    cams = sorted(cams, key=lambda c: c['img_name'])
    test = cams[::TEST_EVERY]
    if MAX_TEST_VIEWS:
        test = test[:MAX_TEST_VIEWS]
    print(f"\n  [G4] {len(test)} held-out views (every {TEST_EVERY}th of "
          f"{len(cams)})")

    acc = {name: {'psnr': [], 'ssim': [], 'lpips': []} for name in maps}
    for k, cam in enumerate(test):
        s = make_sensor_fn(cam)
        gt = None
        for name, model in maps.items():
            n = 256 if name in STOCHASTIC else 1
            img, _ = _render_timed(s, model, n)
            pred = np.clip(img.cpu().numpy(), 0, 1).astype(np.float32)
            if gt is None:
                gt = _load_gt_image(images_dir, cam['img_name'], pred.shape[:2])
                if gt is None:
                    print(f"    view {k}: no photograph for "
                          f"{cam['img_name']}, skipping")
                    break
            p, ss, lp = _metrics(pred, gt)
            acc[name]['psnr'].append(p)
            if ss is not None: acc[name]['ssim'].append(ss)
            if lp is not None: acc[name]['lpips'].append(lp)
        print(f"    view {k + 1}/{len(test)} done", end="\r")

    def mean(v):
        return f"{np.mean(v):.4f}" if v else "n/a"
    rows = [[name.replace('_', '\\_'),
             256 if name in STOCHASTIC else 1,
             mean(acc[name]['psnr']), mean(acc[name]['ssim']),
             mean(acc[name]['lpips'])] for name in maps]
    save_table("G4_held_out",
               ["Method", "spp", "PSNR $\\uparrow$", "SSIM $\\uparrow$",
                "LPIPS $\\downarrow$"], rows,
               f"Novel-view synthesis quality on {len(test)} held-out views. "
               "Deterministic methods are exact at one sample per pixel.",
               "held_out")
    return rows


# ==================================================================

def run_all(maps=None, sensor=None, crop=None, make_sensor_fn=None,
            gt_camera=None):
    """maps          {'GS3D_Sampled': m, 'DSYG': m, 'Sun': m}
    sensor        the sensor used for the timing/convergence/figure runs
    crop          (y0, y1, x0, x1) detail region, or None
    make_sensor_fn  your notebook's make_sensor(cam_dict) -> sensor, needed
                  for the held-out metrics
    gt_camera     the cameras.json entry matching `sensor`, so the comparison
                  figure can include the photograph alongside the renders"""
    _ensure_dir()
    crop = crop if crop is not None else DETAIL_CROP

    print("=" * 70)
    print("PART A -- analytic experiments (no GPU required)")
    print("=" * 70)
    A1_overlap_accuracy()
    A2_separation_sweep()
    A3_opacity_sweep()
    A4_depth_bias()
    A5_numerical_stability()

    if maps is None or sensor is None:
        print("\n" + "=" * 70)
        print("PART B skipped -- pass maps={...} and sensor=... to run it.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("PART B -- GPU experiments")
        print("=" * 70)
        G1_render_time(maps, sensor)
        G2_convergence(maps, sensor, crop)

        gt = None
        if gt_camera is not None and IMAGES_DIR:
            probe, _ = _render_timed(sensor, list(maps.values())[0], 1)
            gt = _load_gt_image(IMAGES_DIR, gt_camera['img_name'],
                                probe.cpu().numpy().shape[:2])
            if gt is None:
                print("  [G3] no photograph found for gt_camera; "
                      "figure will show renders only.")
        G3_qualitative(maps, sensor, crop, gt=gt)
        G4_held_out(maps, make_sensor_fn=make_sensor_fn)

    print(f"\nAll outputs written to {os.path.abspath(RESULTS_DIR)}")
    print("  .csv  raw numbers")
    print("  .tex  paste-ready booktabs tables")
    print("  .png  figures")


if __name__ == "__main__":
    run_all()
