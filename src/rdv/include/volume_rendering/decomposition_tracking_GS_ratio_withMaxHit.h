/*
Ratio-Tracking-style Decomposition Tracking for 3D Gaussian Splatting
Based on: "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"
          (Kutz, Habel, Li, Novak 2017) -- ratio-tracking branch.

CONTRAST WITH decomposition_tracking_GS.h (delta-tracking-style):
  - delta-tracking-style: one stochastic dice roll per candidate; the first
    accepted hit wins and the ray stops; unbiased only after averaging many
    independent traces (many samples-per-pixel).
  - ratio-tracking-style (this file): every candidate intersection is kept,
    weighted deterministically by its true-depth transmittance, and all of
    them are accumulated in a single traversal. There is no accept/reject
    randomness in the blending step itself.

READ BEFORE TRUSTING THIS FILE:
The correct weight for intersection i is
    T_i = product over every OTHER intersection j strictly closer to the
          camera than i, of (1 - alpha_j).
That is inherently an order-dependent quantity -- computing it exactly
requires knowing, for every pair (i, j), which one is closer. This file
computes T_i EXACTLY via a direct O(hit_count^2) pairwise depth comparison
over the *unsorted* hit list gathered by the ray query, rather than via an
explicit comparison-sort (DSYG's approach). That avoids a sort, but it is a
different way of resolving the same ordering information, not a free lunch:
for rays with many overlapping Gaussians it can cost more than DSYG's
O(k log k) sort + O(k) blend. What it buys you is an estimator with (up to
the MAX_HITS cap and floating point) EXACTLY zero compositing-side variance,
which is the right reference point for "ratio tracking in the limit."

This is deliberately NOT a cheaper stochastic ratio-tracking variant that
walks candidates in whatever order the BVH traversal happens to discover
them -- doing that would introduce a bias controlled by how well BVH
discovery order tracks true depth order, which needs to be characterized
empirically (and discussed with your advisor) before it should be presented
as "ratio tracking" in the thesis.

Overflow policy for hit_count > MAX_HITS matches DSYG exactly (evict the
current farthest candidate when a closer one arrives), so that any
difference you measure between this file and DSYG is not an artifact of a
different truncation policy.
*/

FORWARD {
    // ---- BINDINGS ----
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

    // ---- RAY ----
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = normalize(vec3(_input[3], _input[4], _input[5]));

    // ---- SH (3DGS convention: camera-to-point direction = w) ----
    float sh_coefs[16];
    eval_sh(w, sh_coefs);

    // ---- PHASE 1: COLLECT every candidate + its exact alpha (unsorted) ----
    const int MAX_HITS = 4096;
    int   hit_indices[MAX_HITS];
    float hit_depths[MAX_HITS];
    float hit_alphas[MAX_HITS];
    int   hit_count = 0;

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, accelerationStructureEXT(parameters.ads),
        gl_RayFlagsOpaqueEXT, 0xFF, x, 0.0, w, 10000.0);

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
                float t_peak = -B / A;
                if (t_peak > 0.0) {
                    float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z
                            + 2.0*(M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);
                    float power = -0.5 * (C - (B*B)/A);

                    if (power > -15.0 && power <= 0.0) {
                        float target_alpha = min(opacities.data[i], 0.999);
                        float peak_tau  = -log(1.0 - target_alpha);
                        float exact_tau = peak_tau * exp(power);
                        float alpha     = 1.0 - exp(-exact_tau);

                        // matches DSYG's negligible-contribution cutoff (1/255)
                        if (alpha >= 0.0039) {
                            if (hit_count < MAX_HITS) {
                                hit_indices[hit_count] = i;
                                hit_depths[hit_count]  = t_peak;
                                hit_alphas[hit_count]  = alpha;
                                hit_count++;
                            } else {
                                // same overflow policy as DSYG: keep the closest
                                // MAX_HITS, replace the current farthest one
                                int   max_idx = 0;
                                float max_d   = hit_depths[0];
                                for (int j = 1; j < MAX_HITS; ++j)
                                    if (hit_depths[j] > max_d) { max_d = hit_depths[j]; max_idx = j; }
                                if (t_peak < max_d) {
                                    hit_indices[max_idx] = i;
                                    hit_depths[max_idx]  = t_peak;
                                    hit_alphas[max_idx]  = alpha;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- PHASE 2: WEIGHT every intersection by its exact true-depth transmittance ----
    // T_i = product, over all j with depth_j < depth_i, of (1 - alpha_j).
    // Resolved by direct pairwise depth comparison -- no comparison-sort of the array.
    vec3 final_color = vec3(0.0);

    for (int i = 0; i < hit_count; ++i) {
        float T_i = 1.0;
        for (int j = 0; j < hit_count; ++j) {
            if (j != i && hit_depths[j] < hit_depths[i]) {
                T_i *= (1.0 - hit_alphas[j]);
                if (T_i < 0.001) break;  // occluded regardless of remaining j's
            }
        }
        if (T_i < 0.001) continue;  // fully occluded: skip its color evaluation

        int pi = hit_indices[i];
        vec3 gaussian_color = colors.data[pi] * sh_coefs[0];
        int base = pi * 45;
        for (int c = 1; c < 16; ++c) {
            gaussian_color.x += f_rest.data[base + (c-1)     ] * sh_coefs[c];
            gaussian_color.y += f_rest.data[base + (c-1) + 15] * sh_coefs[c];
            gaussian_color.z += f_rest.data[base + (c-1) + 30] * sh_coefs[c];
        }
        gaussian_color = clamp(gaussian_color + 0.5, 0.0, 1.0);

        final_color += T_i * hit_alphas[i] * gaussian_color;
    }

    _output = float[](final_color.x, final_color.y, final_color.z);
}
BACKWARD {}
