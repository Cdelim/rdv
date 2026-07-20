/*
Don't Splat Your Gaussians -- deterministic sort-and-blend baseline
UNCAPPED, TRANSMITTANCE-TERMINATED variant (v2 -- keep the original
DontSplashYourGaussians.h alongside this one; the capped-vs-uncapped
difference is itself a legitimate thing to report, not just a bug fix).

THE PROBLEM WITH THE ORIGINAL: it collected up to MAX_HITS=256 candidates
(evicting the farthest on overflow), THEN sorted, THEN accumulated with an
early break once T<0.001. Two failure modes: in a scene with more than 256
meaningfully-overlapping Gaussians along one ray -- exactly the near-opaque
regime this thesis is about -- it silently truncates. In the opposite case
(density low enough that T dies out well under 256 hits), it still
exhaustively collects and sorts everything the BVH finds along the entire
ray, doing real work on candidates far behind the point where they could
possibly still matter.

WHY YOU CAN'T JUST DELETE THE BUFFER AND BREAK EARLY ON A SINGLE PASS:
transmittance-based early termination is only valid when accumulating in
TRUE front-to-back depth order. rayQueryProceedEXT delivers AABB candidates
in BVH TRAVERSAL order, which is not guaranteed to be depth order. A single
pass that tries to accumulate-and-break as candidates arrive could
terminate having missed a closer candidate the traversal simply hadn't
reached yet -- silently wrong, not just imprecise.

THE FIX: repeated nearest-hit queries. Each outer iteration launches a
FRESH ray query with tMin advanced to just past the last accepted hit,
scans every candidate that pass finds, and keeps only the single closest
one with depth > tMin. Because every candidate in a given pass genuinely IS
compared against every other in that same pass, this is correct regardless
of BVH traversal order -- validated numerically (200 random trials, shuffled
discovery order, exact match to a correctly-sorted reference; see the
Python simulation this was checked against before being ported here). The
outer loop stops the moment T drops below threshold, or the ray runs out of
geometry. No array. No cap tied to scene density.

COST TRADE-OFF, stated plainly: this re-traverses (parts of) the BVH once
per ACCEPTED hit rather than once per ray, so for a ray that only grazes a
handful of widely-separated Gaussians it costs more than collect-once-then-
sort did. It should win specifically in the dense, near-opaque regime this
thesis cares about -- the numerical check above found it correctly stopping
after 14 iterations on a 200-candidate dense scene, versus the old approach
needing to sort all 200 first to discover the same thing -- but it is not
strictly cheaper in every scene, and that's worth stating in the thesis
rather than presenting this as an unqualified win.

SAFETY_MAX_ITERS below is NOT a quality cap like the old MAX_HITS -- it's a
defensive bound against a shader hang if something upstream is broken
(e.g. many zero-thickness Gaussians at identical depth preventing t_cursor
from ever advancing). Set high enough that no realistic scene should ever
reach it; hitting it is a bug to find, not a scene to work around.

NOTE ON decomposition_tracking_GS_ratio.h: once that file uses this exact
same pattern (see its own v2), it computes the identical quantity this file
does, by the identical method. That's expected, not a mistake -- see that
file's header for what actually still differs between the delta-tracking
estimator (GS3D) and this deterministic target.
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

    const int SAFETY_MAX_ITERS = 4096;
    for (int iter = 0; iter < SAFETY_MAX_ITERS; ++iter) {
        if (T < 0.001) break;

        // --- one full pass: find the SINGLE closest candidate with t > t_cursor ---
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
                    // only candidates strictly beyond where we've already
                    // accounted for, and only keep it if it's the closest
                    // such candidate seen in THIS pass
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

        if (best_idx < 0) break;  // ray has exited all geometry

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
        t_cursor = best_t + 1e-4;  // nudge past this hit so the next pass can't re-find it
    }

    _output = float[](final_color.x, final_color.y, final_color.z);
}
BACKWARD {
    // Differentiation logic goes here
}