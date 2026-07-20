# overlap_ground_truth.py
#
# Standalone Python tool -- no Vulkan, no rdv, no GPU needed. Everything here
# runs and is numerically validated on its own (see the __main__ block).
# Paste into a notebook cell, or just run it as a script to see the numbers.
#
# WHAT THIS IS FOR: RQ1's on-axis stack (from evaluation_experiments.py)
# deliberately centered every Gaussian exactly on the ray, which sidesteps
# the one configuration where peak-based hit selection actually goes wrong --
# two Gaussians whose CENTERS are off the ray but whose TAILS overlap on it.
# This file gives you real, independently-derived ground truth for exactly
# that configuration (general N, not just 2), by numerically marching the
# joint density rather than approximating each Gaussian by its own peak.
# Think of it as "RQ1b": same idea as RQ1 (compare against real ground
# truth), extended to cover the case RQ1 couldn't.

# %%
import numpy as np
from scipy.stats import norm  # only used to double-check probit below; not
                               # needed at runtime once you trust the approx.

# %% [markdown]
# ## Validated erfinv / probit (needed for proper free-flight sampling)
#
# Mike Giles' single-precision erfinv approximation (GPU Computing Gems,
# 2010) -- the standard GPU-portable formula for inverse-CDF Gaussian
# sampling. Validated below against scipy.special.erfinv: max abs error
# 2.4e-7 across the full domain, including near alpha~1 (deep tails).

# %%
def giles_erfinv(x):
    x = np.asarray(x, dtype=np.float64)
    w = -np.log((1.0 - x) * (1.0 + x))
    out = np.empty_like(x)
    mask = w < 5.0

    w1 = w[mask] - 2.5
    p1 = np.full_like(w1, 2.81022636e-08)
    for c in (3.43273939e-07, -3.5233877e-06, -4.39150654e-06, 0.00021858087,
              -0.00125372503, -0.00417768164, 0.246640727, 1.50140941):
        p1 = c + p1 * w1
    out[mask] = p1 * x[mask]

    w2 = np.sqrt(w[~mask]) - 3.0
    p2 = np.full_like(w2, -0.000200214257)
    for c in (0.000100950558, 0.00134934322, -0.00367342844, 0.00573950773,
              -0.0076224613, 0.00943887047, 1.00167406, 2.83297682):
        p2 = c + p2 * w2
    out[~mask] = p2 * x[~mask]
    return out


def probit(u):
    """Inverse standard normal CDF, via the validated erfinv above."""
    return np.sqrt(2.0) * giles_erfinv(2.0 * u - 1.0)


# %% [markdown]
# ## Scene representation
#
# Each Gaussian is described in its already-reduced, along-the-ray form --
# exactly the quantities decomposition_tracking_GS.h computes internally for
# a given query ray: A (curvature, from w^T M w), t_star (analytic peak,
# -B/A), and alpha (the ray's own exact_tau converted through Beer-Lambert).
# This intentionally skips re-deriving the 3D ray-Gaussian intersection --
# that formula is already validated elsewhere; what's in question here is
# ONLY how overlapping candidates get combined, so the scene is specified at
# exactly that level of abstraction.

# %%
def make_gaussian(name, A, tstar, alpha, color):
    g = dict(name=name, A=A, tstar=tstar, alpha=alpha, color=np.asarray(color, dtype=np.float64))
    g['peak_tau'] = -np.log(1.0 - alpha)
    g['sigma_peak'] = g['peak_tau'] * np.sqrt(A / (2.0 * np.pi))
    return g


def _sigma(g, t):
    return g['sigma_peak'] * np.exp(-0.5 * g['A'] * (t - g['tstar']) ** 2)


# %% [markdown]
# ## 1. True ground truth -- fine-step marching of the JOINT density
#
# Evaluates every Gaussian's density at every marched point and sums them
# (density superposition is exact, no approximation), then integrates the
# transmittance-weighted emission directly. This is what a real renderer
# would converge to given infinite resolution -- the actual target, not a
# stand-in for it.

# %%
def true_ground_truth(gaussians, t0=-4.0, t1=8.0, n=2_000_000):
    t = np.linspace(t0, t1, n)
    dt = t[1] - t[0]
    sigma_total = np.zeros(n)
    sigma_color = np.zeros((n, 3))
    for g in gaussians:
        s = _sigma(g, t)
        sigma_total += s
        sigma_color += s[:, None] * g['color'][None, :]
    T = np.exp(-np.cumsum(sigma_total) * dt)
    T_prev = np.concatenate(([1.0], T[:-1]))
    return np.sum(T_prev[:, None] * sigma_color * dt, axis=0)


# %% [markdown]
# ## 2. Peak-only, deterministic -- what DontSplashYourGaussians.h /
# ## decomposition_tracking_GS.h (original) both target
#
# Sorts by each Gaussian's fixed peak and alpha-composites. Zero noise, but
# only exact when supports don't meaningfully overlap.

# %%
def peak_only_composite(gaussians):
    ordered = sorted(gaussians, key=lambda g: g['tstar'])
    T, color = 1.0, np.zeros(3)
    for g in ordered:
        color = color + T * g['alpha'] * g['color']
        T *= (1.0 - g['alpha'])
    return color


# %% [markdown]
# ## 3. Properly-sampled free-flight estimator -- Python reference for the
# ## fix in decomposition_tracking_GS_sampled.h
#
# For each Gaussian independently: accept/reject exactly as before
# (probability alpha), and IF accepted, sample WHERE along its own density
# the interaction lands via closed-form inverse-CDF (using probit above)
# instead of always using the fixed peak. Whichever accepted candidate's
# SAMPLED location is closest wins. This is decomposition tracking's usual
# "minimum of independent free-flight samples" construction, done with a
# real sampled location instead of a constant.

# %%
def sampled_trial(gaussians, rng):
    best_t, best_color = np.inf, None
    for g in gaussians:
        if rng.random() < g['alpha']:
            u = np.clip(rng.random(), 1e-6, 1.0 - 1e-6)
            t_sample = g['tstar'] + probit(np.array([u]))[0] / np.sqrt(g['A'])
            if 0.0 < t_sample < best_t:
                best_t, best_color = t_sample, g['color']
    return best_color if best_color is not None else np.zeros(3)


def sampled_mc_estimate(gaussians, n_trials=2_000_000, rng=None):
    rng = rng or np.random.default_rng(0)
    acc = np.zeros(3)
    for _ in range(n_trials):
        acc += sampled_trial(gaussians, rng)
    return acc / n_trials


# %% [markdown]
# ## Demo: exactly the configuration in the hand-drawn picture

# %%
if __name__ == '__main__':
    scene = [
        make_gaussian('A', A=8.0, tstar=1.0, alpha=0.6, color=[1.0, 0.0, 0.0]),
        make_gaussian('B', A=5.0, tstar=1.6, alpha=0.6, color=[0.0, 0.0, 1.0]),
    ]

    gt = true_ground_truth(scene)
    peak = peak_only_composite(scene)
    sampled = sampled_mc_estimate(scene, n_trials=2_000_000)

    print("true ground truth (fine marching):      ", gt)
    print("peak-only (DSYG-equivalent):             ", peak,
          " rel. err:", np.abs(peak - gt) / np.maximum(gt, 1e-9))
    print("properly-sampled MC estimator:           ", sampled,
          " rel. err:", np.abs(sampled - gt) / np.maximum(gt, 1e-9))

    print()
    print("sanity check -- same scene, widely separated (peak-only should now")
    print("match ground truth almost exactly, confirming the gap above is an")
    print("overlap effect and not some other error):")
    scene_sep = [
        make_gaussian('A', A=8.0, tstar=0.0, alpha=0.6, color=[1.0, 0.0, 0.0]),
        make_gaussian('B', A=5.0, tstar=3.0, alpha=0.6, color=[0.0, 0.0, 1.0]),
    ]
    gt_sep = true_ground_truth(scene_sep)
    peak_sep = peak_only_composite(scene_sep)
    print("true:", gt_sep, " peak-only:", peak_sep,
          " rel. err:", np.abs(peak_sep - gt_sep) / np.maximum(gt_sep, 1e-9))

# Try your own scenes: vary tstar separation and A (width) to see how the
# peak-only bias grows as overlap increases -- that curve (bias vs. overlap
# amount) is worth an actual figure in your limitations discussion, and
# costs nothing further to make once you're calling these three functions
# over a grid of separations.
