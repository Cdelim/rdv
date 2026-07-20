/*
Ratio-Tracking-style Decomposition Tracking for 3D Gaussian Splatting
UNCAPPED, TRANSMITTANCE-TERMINATED variant (v2)

THIS SUPERSEDES decomposition_tracking_GS_ratio.h, not just its MAX_HITS
cap. The earlier version resolved depth order by collecting up to 256
candidates and comparing every pair (O(hit_count^2)) specifically to avoid
an explicit sort. Once you also need "no fixed cap, terminate on T", that
O(k^2) approach doesn't extend cleanly -- you'd either need an even bigger
fixed array (still a cap, just a higher one) or some way to know you can
stop collecting before you've seen everything, which you can't safely know
from an unsorted partial scan. Repeated nearest-hit queries (below) solve
both problems at once, so this file replaces the pairwise-comparison
approach entirely rather than patching it.

BE AWARE OF WHAT THIS MEANS: with this fix in place,
decomposition_tracking_GS_ratio.h and DontSplashYourGaussians_v2.h compute
the EXACT SAME quantity, by the EXACT SAME method (repeated nearest-hit,
front-to-back accumulation, transmittance termination). This isn't a new
coincidence introduced by this rewrite -- "ratio tracking, done exactly"
was always mathematically identical to sorted deterministic alpha
compositing; the two files previously differed only in HOW they resolved
depth order (O(k^2) pairwise comparison here vs. an explicit sort there),
not in WHAT they computed. Now they don't even differ in that. Worth
deciding deliberately, for the thesis, whether to:
  (a) keep both files for narrative clarity (this one represents the
      decomposition-tracking framing your professor asked for; DSYG_v2
      represents the literature baseline reproduction) even though they're
      now the same algorithm under different names, and say so explicitly
      in the Method chapter, or
  (b) drop this file and just use DontSplashYourGaussians_v2.h as the one
      deterministic reference, since maintaining two identical shaders
      under different names has no empirical payoff.
Either is defensible; what isn't defensible is presenting a
GS3D-vs-GS3D_Ratio comparison as if it tests something different from
GS3D-vs-DSYG, now that both reference points are the same number.

What the REAL remaining three-way comparison in this thesis is:
  - GS3D            (delta-tracking-style: stochastic, single accepted hit
                      per trace, needs many traces to converge)
  - GS3D_Sampled     (delta-tracking-style, but with a properly sampled hit
                      location instead of always the fixed peak -- see its
                      own header for why that matters for overlapping
                      Gaussians specifically)
  - DSYG / GS3D_Ratio (deterministic, zero compositing-side variance,
                      the target both stochastic estimators should converge
                      toward)
That's still a genuine, useful comparison -- it's just a two-estimator
comparison against one deterministic reference, not three independent
things.
*/

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

    float sh_coefs[16];
    eval_sh(w, sh_coefs);

    float T = 1.0;
    vec3 final_color = vec3(0.0);
    float t_cursor = 0.0;

    const int SAFETY_MAX_ITERS = 4096;  // defensive bound, not a quality cap -- see DSYG_v2 header
    for (int iter = 0; iter < SAFETY_MAX_ITERS; ++iter) {
        if (T < 0.001) break;

        int   best_idx   = -1;
        float best_t     = 1e30;
        float best_alpha = 0.0;

        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, accelerationStructureEXT(parameters.ads),
            gl_RayFlagsOpaqueEXT, 0xFF, x, t_cursor, w, 10000.0);

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
                float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                        + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                        + M12*(w.y*d.z+w.z*d.y);

                if (A > 1e-6) {
                    float t_star = -B / A;
                    if (t_star > t_cursor && t_star < best_t) {
                        float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z
                                + 2.0*(M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);
                        float power = -0.5 * (C - (B*B)/A);
                        if (power > -15.0 && power <= 0.0) {
                            float target_alpha = min(opacities.data[i], 0.999);
                            float peak_tau  = -log(1.0 - target_alpha);
                            float exact_tau = peak_tau * exp(power);
                            float alpha     = 1.0 - exp(-exact_tau);
                            if (alpha >= 0.0039) {
                                best_idx = i; best_t = t_star; best_alpha = alpha;
                            }
                        }
                    }
                }
            }
        }

        if (best_idx < 0) break;

        vec3 gaussian_color = colors.data[best_idx] * sh_coefs[0];
        int rest_idx = best_idx * 45;
        for (int c = 1; c < 16; ++c) {
            gaussian_color.x += f_rest.data[rest_idx + (c-1)     ] * sh_coefs[c];
            gaussian_color.y += f_rest.data[rest_idx + (c-1) + 15] * sh_coefs[c];
            gaussian_color.z += f_rest.data[rest_idx + (c-1) + 30] * sh_coefs[c];
        }
        gaussian_color = clamp(gaussian_color + 0.5, 0.0, 1.0);

        final_color += T * best_alpha * gaussian_color;
        T *= (1.0 - best_alpha);
        t_cursor = best_t + 1e-4;
    }

    _output = float[](final_color.x, final_color.y, final_color.z);
}
BACKWARD {
    // Differentiation logic goes here
}