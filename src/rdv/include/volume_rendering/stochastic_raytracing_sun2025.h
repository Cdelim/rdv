/*
Stochastic Ray Tracing of Transparent 3D Gaussians -- Sun, Georgiev, Fei, Hašan
(Adobe), Eurographics Symposium on Rendering 2025, arXiv:2504.06598v3.

Faithful reproduction of their Algorithm 1 in this project's Vulkan ray-query
framework, for use as an EVALUATION BASELINE. This is the closest published
prior work to this thesis's delta-tracking-style estimator: single BVH
traversal, per-intersection Russian Roulette on opacity, closest accepted
intersection shaded, no sorting, proven unbiased (their Eq. 5).

=============================================================================
HOW THIS DIFFERS FROM decomposition_tracking_GS.h / _sampled.h (READ FIRST)
=============================================================================

1. SHADING POSITION -- the key scientific difference.
   Sun et al. use the MEAN of the 1D Gaussian along the ray (their Sec. 3.1,
   Alg. 1 line 3: `t <- g1.mu`), i.e. the analytic peak t* = -B/A. Their
   `OursCenter` variant instead projects the Gaussian CENTER onto the camera
   direction, for compatibility with rasterizer-trained assets (their Sec.
   4.1). NEITHER samples a position from the Gaussian's own density.
   decomposition_tracking_GS_sampled.h does sample that position, which on
   the validated 2-Gaussian overlap scene used throughout this project was
   the difference between ~10-18% relative error (peak-only) and 0.1-0.4%
   (sampled) against numerically-marched ground truth. So this file is
   expected to reproduce the peak-only bias -- that is the point of having it
   as a baseline, not a defect in the reproduction.

2. RAY CLIPPING -- their main performance optimization, which this project's
   own shaders do NOT currently do. On acceptance, Alg. 1 line 15 reports the
   hit and clips the ray: `r.tmax <- t`, so BVH nodes beyond the accepted hit
   are skipped for the remainder of the traversal. Implemented here with
   rayQueryGenerateIntersectionEXT(), which commits an AABB hit and shrinks
   the query's t-range. GS3D instead tracks `closest_t` manually and
   traverses the full ray range regardless -- correct, but strictly more work.

3. RNG. They use a STATELESS position-dependent trigonometric hash (their
   Sec. 4.2, Eqs. 8-10) rather than a stateful per-ray generator, because
   Vulkan/DXR forbid writing to the ray payload from an intersection shader.
   That constraint does not bind here -- this is a compute shader using ray
   QUERIES, so a stateful `random()` is available -- but the hash is
   reproduced faithfully since it changes the noise characteristics (and
   their reported results). Validated numerically before porting: over 200k
   samples in a [-3,3]^3 region the hash gave mean 0.4985, std 0.2891, and
   chi-square 12.9 against uniform over 10 bins (df=9; below the 21.7
   rejection threshold at p=0.01), i.e. adequately uniform for this use.
   NOTE: GLSL fract() returns a value in [0,1) even for negative inputs
   (fract(x) = x - floor(x)), which is the behavior this code relies on.

4. SUPPORT RADIUS. They use s = 2*sqrt(2) ~= 2.83 standard deviations (their
   Eq. 2 and Sec. 3.1), vs. this project's 3.0 elsewhere. Since the
   Mahalanobis distance squared at the peak equals -2*power, their
   negligibility test (Alg. 1 line 7-8) is exactly `power >= -s^2/2 = -4.0`,
   noticeably tighter than the `power > -15.0` cutoff used in this project's
   other shaders. Kept faithful to theirs here.
   The AABBs themselves come from the acceleration structure built by the
   Python wrapper (3-sigma), so this only affects the per-candidate cull, not
   the BVH -- a minor, documented inconsistency with their setup.

5. ALPHA. alpha = opacity * exp(power), the standard 3DGS convention (same as
   DontSplashYourGaussians_v5_official.h), NOT the optical-depth
   reinterpretation used by GS3D/DSYG_v2.

-----------------------------------------------------------------------------
A LIKELY TYPO IN THEIR PAPER, and what this file does about it:
Algorithm 1 lines 12-13 read `if xi < g1.alpha then return  [rejected by
Russian Roulette]` -- i.e. reject when xi < alpha, accepting with probability
1-alpha. That contradicts both their Eq. (4) (`alpha_hat = 1 with probability
alpha`) and their Fig. 2 caption (`alpha_hat_i = 1 if xi_i < alpha_i`), and
would inverse-weight the whole image. This file implements the version
consistent with Eq. 4 and Fig. 2: ACCEPT when xi < alpha. Flagging rather
than silently "fixing" it, since it affects how you describe the baseline.

-----------------------------------------------------------------------------
MULTI-SAMPLE (their Sec. 3.5 / Eq. 7): their Eq. 7 averages N independent
evaluations, and they note explicitly that "this can be achieved by tracing a
ray independently N times." That is EXACTLY what this framework's
`sensor.view(model, samples=N)` already does, so Eq. 7 comes for free -- no
shader work needed. Their single-traversal N-instantiation trick is a pure
performance optimization of the same estimator, not a different estimator,
and is deliberately not implemented here to keep this file simple and
verifiable.
=============================================================================

Set DEPTH_MODE below: 0 = OursMean (1D Gaussian mean / analytic peak),
1 = OursCenter (projected Gaussian center; their Sec. 4.1, closer to how
rasterizer-trained assets like the 3DGS bicycle scene were optimized, and
what their Fig. 3 shows scoring better on such assets).
*/

#define DEPTH_MODE 0   // 0 = OursMean, 1 = OursCenter

// Their Eqs. 8-10, constants from Sec. 4.2.
float sun_r1(float q) {
    return fract(47453.5453 * sin(91.3458 * q));
}

float sun_r2(vec2 q) {
    return fract(43758.5453 * sin(dot(q, vec2(12.9898, 78.233))));
}

// xi(p) = r2(p.xy + r1(p.z)) -- their Eq. 10
float sun_hash(vec3 p) {
    return sun_r2(p.xy + vec2(sun_r1(p.z)));
}

FORWARD {
    GPUPtr positions_ptr = load_tensor(parameters.positions);
    vec3_ptr positions = vec3_ptr(positions_ptr);
    GPUPtr colors_ptr = load_tensor(parameters.colors);
    vec3_ptr colors = vec3_ptr(colors_ptr);
    GPUPtr inv_covs_ptr = load_tensor(parameters.inv_covs);
    float_ptr inv_covs = float_ptr(inv_covs_ptr);
    GPUPtr opacities_ptr = load_tensor(parameters.opacities);
    float_ptr opacities = float_ptr(opacities_ptr);
    GPUPtr f_rest_ptr = load_tensor(parameters.f_rest);
    float_ptr f_rest = float_ptr(f_rest_ptr);

    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = normalize(vec3(_input[3], _input[4], _input[5]));

    // Their hash is position-dependent and must still decorrelate across the
    // sensor's samples-per-pixel (their Sec. 4.2 perturbs the hit position
    // with a frame-dependent quasi-random offset for exactly this reason).
    // This framework's stateful random() supplies that per-sample offset.
    vec3 jitter = vec3(random(), random(), random()) * 1024.0;

    float sh_coefs[16];
    eval_sh(w, sh_coefs);

    // Their Eq. 2 support radius: s = 2*sqrt(2), so Mahalanobis^2 <= 8,
    // and since Mahalanobis^2 at the peak == -2*power, this is power >= -4.
    const float POWER_CUTOFF = -4.0;

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, accelerationStructureEXT(parameters.ads),
        gl_RayFlagsNoneEXT, 0xFF, x, 0.0, w, 10000.0);

    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) ==
            gl_RayQueryCandidateIntersectionAABBEXT) {

            int i = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            int cov_idx = i * 6;
            vec3 d = x - positions.data[i];

            float M00 = inv_covs.data[cov_idx + 0];
            float M01 = inv_covs.data[cov_idx + 1];
            float M02 = inv_covs.data[cov_idx + 2];
            float M11 = inv_covs.data[cov_idx + 3];
            float M12 = inv_covs.data[cov_idx + 4];
            float M22 = inv_covs.data[cov_idx + 5];

            float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                    + 2.0*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
            if (A <= 1e-6) continue;
            float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                    + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                    + M12*(w.y*d.z+w.z*d.y);
            float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z
                    + 2.0*(M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);

            // --- Alg. 1 line 2-3: 1D Gaussian along the ray, and its mean ---
            float t_peak = -B / A;
            float power  = -0.5 * (C - (B*B)/A);

            // --- Alg. 1 lines 7-9: negligibility cull (their Eq. 2) ---
            // Evaluated at the peak in BOTH depth modes, per Alg. 1 line 8
            // ("mean of g1 is outside the AABB of g").
            if (power < POWER_CUTOFF) continue;

            // --- depth used for ordering / reporting ---
#if DEPTH_MODE == 0
            float t_hit = t_peak;                 // OursMean
#else
            float t_hit = dot(-d, w);             // OursCenter (their Sec. 4.1):
                                                   // project center onto ray dir
#endif

            // --- Alg. 1 lines 4-6: valid ray range ---
            // rayQueryGenerateIntersectionEXT keeps shrinking tmax as hits are
            // accepted, so re-checking against the CURRENT range is what makes
            // the clipping optimization actually pay off.
            if (t_hit <= 0.0) continue;
            if (t_hit >= rayQueryGetIntersectionTEXT(rq, true) &&
                rayQueryGetIntersectionTypeEXT(rq, true) !=
                    gl_RayQueryCommittedIntersectionNoneEXT) continue;

            // --- Alg. 1 lines 10-14: Russian Roulette on opacity ---
            vec3 p_hit = x + t_hit * w;
            float xi = sun_hash(p_hit + jitter);
            float alpha = min(opacities.data[i] * exp(power), 0.9999);

            // ACCEPT if xi < alpha -- per their Eq. 4 and Fig. 2.
            // (Their Alg. 1 line 12 reads inverted; see header note.)
            if (xi < alpha) {
                // --- Alg. 1 line 15: report + clip ray (r.tmax <- t) ---
                rayQueryGenerateIntersectionEXT(rq, t_hit);
            }
        }
    }

    // --- shade the single closest accepted intersection (their Eq. 6) ---
    // No alpha and no transmittance multiply here, deliberately: the Russian
    // Roulette acceptance IS the alpha weighting, and "closest accepted wins"
    // IS the transmittance. Multiplying by alpha again would double-count and
    // bias the image dark. (Same reasoning as decomposition_tracking_GS.h.)
    vec3 final_color = vec3(0.0);

    if (rayQueryGetIntersectionTypeEXT(rq, true) ==
        gl_RayQueryCommittedIntersectionGeneratedEXT) {

        int pi = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        vec3 gaussian_color = colors.data[pi] * sh_coefs[0];
        int rest_idx = pi * 45;
        for (int c = 1; c < 16; ++c) {
            gaussian_color.x += f_rest.data[rest_idx + (c-1)     ] * sh_coefs[c];
            gaussian_color.y += f_rest.data[rest_idx + (c-1) + 15] * sh_coefs[c];
            gaussian_color.z += f_rest.data[rest_idx + (c-1) + 30] * sh_coefs[c];
        }
        final_color = clamp(gaussian_color + 0.5, 0.0, 1.0);
    }

    _output = float[](final_color.x, final_color.y, final_color.z);
}
BACKWARD {
    // Not implemented -- forward-only, consistent with this thesis's scope.
}
