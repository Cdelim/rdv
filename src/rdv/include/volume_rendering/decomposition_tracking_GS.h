/*
This map receives a ads data-structure with AABB for each gaussian
parameters:
- ads: Data structure to query the index of the AABB
- positions: each gaussian's position
- cov: each gaussian's covariance
- scale: each gaussian's scale
- sh: each gaussian's spherical harmonic coefficients
*/
/*
Decomposition Tracking for 3D Gaussian Splatting
Based on: "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"
*/

// Helper Function: PCG Hash for Random Number Generation
// Helper Function: PCG Hash for Random Number Generation


FORWARD {
    // 1. MEMORY BINDINGS
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

    // 2. RAY SETUP
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = normalize(vec3(_input[3], _input[4], _input[5]));

    // 3. RNG INIT
    uint b1 = floatBitsToUint(w.x);
    uint b2 = floatBitsToUint(w.y);
    uint b3 = floatBitsToUint(w.z);
    uint seed = b1 ^ (b2 * 1973u) ^ (b3 * 9277u);
    rdv_rng_state = uvec4(seed, seed * 1664525u, ~seed, seed ^ 0x23F1u);
    random_step(); random_step();

    // 4. SH EVALUATION
    // FIX: Use -w (direction from Gaussian toward camera)
    float sh_coefs[16];
    eval_sh(-w, sh_coefs);

    // 5. COLLECT HITS — identical to DSYG so comparison is fair
    const int MAX_HITS = 128;
    int hit_indices[MAX_HITS];
    float hit_depths[MAX_HITS];
    int hit_count = 0;

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
                    if (hit_count < MAX_HITS) {
                        hit_indices[hit_count] = i;
                        hit_depths[hit_count] = t_peak;
                        hit_count++;
                    } else {
                        int max_idx = 0;
                        float max_depth = hit_depths[0];
                        for (int j = 1; j < MAX_HITS; ++j) {
                            if (hit_depths[j] > max_depth) {
                                max_depth = hit_depths[j];
                                max_idx = j;
                            }
                        }
                        if (t_peak < max_depth) {
                            hit_indices[max_idx] = i;
                            hit_depths[max_idx] = t_peak;
                        }
                    }
                }
            }
        }
    }

    // 6. SORT FRONT-TO-BACK
    for (int i = 1; i < hit_count; ++i) {
        int ki = hit_indices[i];
        float kd = hit_depths[i];
        int j = i - 1;
        while (j >= 0 && hit_depths[j] > kd) {
            hit_indices[j+1] = hit_indices[j];
            hit_depths[j+1] = hit_depths[j];
            j--;
        }
        hit_indices[j+1] = ki;
        hit_depths[j+1] = kd;
    }

    // 7. DECOMPOSITION TRACKING — Monte Carlo stochastic estimator
    //
    // KEY IDEA (from Kutz et al.):
    // The alpha compositing sum  SUM_k [ T_{k-1} * alpha_k * color_k ]
    // can be estimated with a SINGLE SAMPLE by walking front-to-back:
    //   - At each Gaussian k, flip a coin with probability alpha_k
    //   - If it lands heads: return color_k (this Gaussian is the "surface")
    //   - If tails: multiply transmittance by (1-alpha_k) and continue
    //
    // The transmittance T_{k-1} is encoded IMPLICITLY:
    // the probability of REACHING Gaussian k equals exactly T_{k-1},
    // because reaching k means all previous coins landed tails.
    // Therefore E[output] = SUM_k T_{k-1} * alpha_k * color_k  (unbiased!)
    //
    // This is the "control component" of decomposition tracking:
    // each Gaussian is treated as a homogeneous slab whose free-path
    // can be sampled analytically from its alpha value.

    vec3 final_color = vec3(0.0);

    for (int h = 0; h < hit_count; ++h) {
        int i = hit_indices[h];
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
        float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z
                + 2.0*(M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);
        float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                + M12*(w.y*d.z+w.z*d.y);

        if (A > 1e-6) {
            float power = -0.5 * (C - (B*B)/A);

            if (power > -15.0 && power <= 0.0) {
                float target_alpha = min(opacities.data[i], 0.999);
                float peak_tau    = -log(1.0 - target_alpha);
                float exact_tau   = peak_tau * exp(power);
                float alpha       = 1.0 - exp(-exact_tau);

                // THE DICE ROLL: heads with probability alpha
                if (random() < alpha) {
                    // This Gaussian is selected as the surface.
                    // NO transmittance multiply needed — T is implicit
                    // in the probability of having reached this Gaussian.
                    vec3 gaussian_color = colors.data[i] * sh_coefs[0];
                    int base = i * 45;
                    for (int c = 1; c < 16; ++c) {
                        gaussian_color.x += f_rest.data[base + (c-1)     ] * sh_coefs[c];
                        gaussian_color.y += f_rest.data[base + (c-1) + 15] * sh_coefs[c];
                        gaussian_color.z += f_rest.data[base + (c-1) + 30] * sh_coefs[c];
                    }
                    final_color = clamp(gaussian_color + 0.5, 0.0, 1.0);
                    break; // done — single sample estimator
                }
                // Tails: this Gaussian is transparent for this ray sample.
                // Transmittance implicitly decreases — next Gaussian is
                // harder to reach (lower prior probability of getting here).
            }
        }
    }

    _output = float[](final_color.x, final_color.y, final_color.z);
}

BACKWARD {}