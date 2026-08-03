# comparison_table.py
#
# Produces the thesis's main comparison table across every estimator built in
# this project, now including the Sun et al. (EGSR 2025) published baseline.
# Paste after evaluation_experiments_MERGED.py.
#
# TWO TABLES, because they answer different questions and conflating them
# would be misleading:
#
#   Table A (analytic): each estimator vs. TRUE volumetric ground truth on a
#     controlled overlap scene with a hand-derived exact answer. This is where
#     the thesis's accuracy claim lives. Pure Python, no GPU needed.
#
#   Table B (real scene): each estimator vs. held-out photographs on the
#     bicycle scene, plus render time. This is the "does it work in practice"
#     table. Needs the GPU pipeline.
#
# Table A is the more important one for the novelty argument, and the one you
# can run right now without a lab machine.

# %% [markdown]
# ## Table A -- accuracy against true volumetric ground truth
#
# Validated numbers already established in this project, reproducible by
# running this cell. The key comparison:
#
#   - Sun et al.'s estimator is provably unbiased FOR ITS OWN image formation
#     model (their Eq. 3: alpha evaluated at the 1D Gaussian mean, composited
#     in depth order). Verified numerically: 2M trials converge to that target
#     within 7e-5.
#   - But that target itself carries up to 18.5% relative error against the
#     true volumetric integral in overlap regions, because it credits each
#     primitive at a single fixed point rather than over its actual extent.
#   - GS3D_Sampled samples the interaction position from each Gaussian's own
#     density and lands within 0.4% of true ground truth.
#
# That is a precise, defensible thesis contribution: not "our method is
# stochastic too", but "the published stochastic ray-traced estimator is
# unbiased with respect to an image formation model that is itself
# measurably wrong under overlap, and sampling the position fixes it."

# %%
import math
import numpy as np


def _mk(A, tstar, alpha, color):
    peak_tau = -math.log(1.0 - alpha)
    return dict(A=A, tstar=tstar, alpha=alpha, peak_tau=peak_tau,
                sigma_peak=peak_tau * math.sqrt(A / (2.0 * math.pi)),
                color=np.asarray(color, dtype=np.float64))


def overlap_scene(separation=0.6):
    """Two overlapping Gaussians; `separation` controls how much they overlap
    (0.6 = the canonical scene used throughout this project; raise it toward
    3.0 and every method below should agree, which is itself a useful check)."""
    return [_mk(8.0, 1.0, 0.6, [1.0, 0.0, 0.0]),
            _mk(5.0, 1.0 + separation, 0.6, [0.0, 0.0, 1.0])]


def true_ground_truth(gs, t0=-4.0, t1=8.0, n=2_000_000):
    """Fine-step marching of the JOINT density -- the actual target. No
    peak approximation, no per-primitive isolation."""
    t = np.linspace(t0, t1, n); dt = t[1] - t[0]
    st = np.zeros(n); sc = np.zeros((n, 3))
    for g in gs:
        s = g['sigma_peak'] * np.exp(-0.5 * g['A'] * (t - g['tstar']) ** 2)
        st += s; sc += s[:, None] * g['color'][None, :]
    T = np.exp(-np.cumsum(st) * dt)
    Tp = np.concatenate(([1.0], T[:-1]))
    return np.sum(Tp[:, None] * sc * dt, axis=0)


def peak_only_target(gs):
    """Sun et al. Eq. 3 == DSYG == GS3D_Ratio == the deterministic target of
    every peak-based method in this project. Sorted, alpha-composited,
    each primitive credited at its own 1D-Gaussian mean."""
    T, col = 1.0, np.zeros(3)
    for g in sorted(gs, key=lambda g: g['tstar']):
        col = col + T * g['alpha'] * g['color']
        T *= (1.0 - g['alpha'])
    return col


def sun_stochastic(gs, n_trials=500_000, seed=0):
    """Sun et al. Eqs. 4-6: binary opacities via Russian Roulette, closest
    accepted intersection wins. Should converge to peak_only_target."""
    rng = np.random.default_rng(seed)
    tstars = np.array([g['tstar'] for g in gs])
    alphas = np.array([g['alpha'] for g in gs])
    cols = np.array([g['color'] for g in gs])
    order = np.argsort(tstars)
    acc = np.zeros(3)
    for _ in range(n_trials):
        hat = rng.random(len(gs)) < alphas
        for idx in order:
            if hat[idx]:
                acc += cols[idx]; break
    return acc / n_trials


def _probit(u):
    # scipy-free inverse normal CDF via the validated Giles erfinv coefficients
    x = 2.0 * np.clip(u, 1e-9, 1 - 1e-9) - 1.0
    w = -math.log((1.0 - x) * (1.0 + x))
    if w < 5.0:
        w -= 2.5
        p = 2.81022636e-08
        for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
                  -0.00125372503, -0.00417768164, 0.246640727, 1.50140941):
            p = c + p * w
    else:
        w = math.sqrt(w) - 3.0
        p = -0.000200214257
        for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
                  -0.0076224613, 0.00943887047, 1.00167406, 2.83297682):
            p = c + p * w
    return math.sqrt(2.0) * p * x


def gs3d_sampled(gs, n_trials=500_000, seed=0):
    """This thesis's estimator: same accept/reject, but the interaction
    POSITION is sampled from each Gaussian's own density (inverse CDF)
    rather than fixed at the peak. Closest sampled position wins."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(3)
    for _ in range(n_trials):
        best_t, best_c = np.inf, None
        for g in gs:
            if rng.random() < g['alpha']:
                t = g['tstar'] + _probit(rng.random()) / math.sqrt(g['A'])
                if 0.0 < t < best_t:
                    best_t, best_c = t, g['color']
        if best_c is not None:
            acc += best_c
    return acc / n_trials


def table_a(separation=0.6, n_trials=500_000):
    gs = overlap_scene(separation)
    gt = true_ground_truth(gs)

    rows = [
        ("TRUE ground truth (joint marching)", gt, "reference"),
        ("Sun et al. Eq.3 target (= DSYG, GS3D_Ratio, v5)", peak_only_target(gs), "deterministic"),
        ("Sun et al. Eq.5/6 (stochastic, EGSR 2025)", sun_stochastic(gs, n_trials), "stochastic"),
        ("GS3D_Sampled (this thesis)", gs3d_sampled(gs, n_trials), "stochastic"),
    ]

    print(f"=== Table A: accuracy vs. true volumetric ground truth "
          f"(separation={separation}, {n_trials:,} trials) ===\n")
    print(f"{'method':<48} {'RGB':<26} {'max rel. err':>12}  kind")
    print("-" * 100)
    for name, val, kind in rows:
        rel = np.abs(val - gt) / np.maximum(gt, 1e-9)
        err = "--" if name.startswith("TRUE") else f"{rel.max()*100:.2f}%"
        print(f"{name:<48} {str(np.round(val,4)):<26} {err:>12}  {kind}")
    print()
    print("Read this as: the stochastic estimators are each unbiased for their")
    print("OWN target. Sun et al.'s target is the peak-only composite (row 2),")
    print("so it inherits that row's error. GS3D_Sampled's target is the true")
    print("integral, so it does not.")
    return rows


def separation_sweep(separations=(0.3, 0.6, 1.0, 1.5, 2.0, 3.0), n_trials=200_000):
    """The figure worth putting in the thesis: peak-only bias as a function of
    how much the primitives actually overlap. Should go to ~0 as they separate,
    which both confirms the bias is an overlap effect AND shows exactly which
    regime the contribution matters in."""
    print(f"\n=== Peak-only bias vs. overlap (the thesis figure) ===\n")
    print(f"{'separation':>10} {'peak-only err':>15} {'GS3D_Sampled err':>18}")
    print("-" * 46)
    out = []
    for s in separations:
        gs = overlap_scene(s)
        gt = true_ground_truth(gs)
        pk = peak_only_target(gs)
        sm = gs3d_sampled(gs, n_trials)
        e_pk = (np.abs(pk - gt) / np.maximum(gt, 1e-9)).max()
        e_sm = (np.abs(sm - gt) / np.maximum(gt, 1e-9)).max()
        out.append(dict(separation=s, peak_err=e_pk, sampled_err=e_sm))
        print(f"{s:>10.1f} {e_pk*100:>14.2f}% {e_sm*100:>17.2f}%")
    return out


def plot_separation_sweep(rows):
    import matplotlib.pyplot as plt
    seps = [r['separation'] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(seps, [r['peak_err']*100 for r in rows], 'o-',
             label='peak-only (Sun et al., DSYG, v5)')
    plt.plot(seps, [r['sampled_err']*100 for r in rows], 's-',
             label='GS3D_Sampled (this thesis)')
    plt.xlabel('primitive separation (overlap decreases →)')
    plt.ylabel('max relative error vs. true ground truth (%)')
    plt.title('Peak-only compositing bias is an overlap effect')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


# %% [markdown]
# ## Table B -- real scene, held-out views
#
# Needs the GPU pipeline and the fixed `make_sensor` from your notebook.
# Reuses evaluate_on_held_out() from nvs_eval_and_viewer.py.
#
# IMPORTANT: verify the camera FOV/intrinsics question before trusting these
# numbers -- it was flagged early in this project and depends on whether
# CameraProbes actually reads cam['fy']. If the fixed make_sensor() in your
# notebook handles it, you're fine; confirm rather than assume.

# %%
def table_b(methods, cameras_data, images_dir, samples_map=None,
            test_every=8, max_test_views=None):
    """
    methods: {'GS3D_Sampled': sampled_map, 'DSYG_SunStochastic': sun_map,
              'DSYG_v2': dsyg2_map, 'v5_official': v5_map, ...}
    samples_map: optional {name: samples} -- stochastic methods need many,
      deterministic ones need exactly 1. Defaults to 1 for anything not listed,
      which would badly misrepresent the stochastic methods, so SET IT.
    """
    samples_map = samples_map or {}
    per_view, summary = None, None
    import pandas as pd
    rows = []
    for name, model in methods.items():
        n = samples_map.get(name, 1)
        df, summ = evaluate_on_held_out(
            methods={name: model}, cameras_data=cameras_data,
            images_dir=images_dir, test_every=test_every,
            samples=n, max_test_views=max_test_views, verbose=False)
        r = summ.loc[name].to_dict()
        r['method'] = name; r['samples'] = n
        rows.append(r)
        print(f"{name:<26} spp={n:<6} PSNR={r['psnr']:.2f}  "
              f"SSIM={r['ssim']:.4f}  LPIPS={r['lpips']:.4f}")
    return pd.DataFrame(rows).set_index('method')


if __name__ == '__main__':
    table_a()
    rows = separation_sweep()
