/*
Don't Splat Your Gaussians -- FAITHFUL implementation (v3)

Unlike DontSplashYourGaussians.h / _v2.h (both evaluate each Gaussian only
at its own isolated peak), this follows the paper's actual Section 4.2
algorithm: partition the ray into segments at every primitive's entry/exit
boundary, walk segments accumulating cumulative transmittance ("regular
tracking" over segments -- their term, and NOT decomposition tracking; the
paper doesn't use decomposition tracking at all), and within whichever
segment a single upfront random draw lands in, solve for the exact hit
location -- closed form if exactly one primitive is active there, BISECTION
if more than one, exactly as the paper states explicitly for the
overlapping case ("we revert to using the bisection solver").

VALIDATED IN PYTHON FIRST (segment_bisection_reference.py) before being
ported here, on the same overlap scene used throughout this thesis: this
algorithm converged to [0.5455, 0, 0.2921], against true marching ground
truth [0.5454, 0, 0.2946] and GS3D_Sampled's [0.5459, 0, 0.2935] -- all
three agree within Monte Carlo noise.

WHAT THAT AGREEMENT MEANS, PLAINLY: this is a heavier, more complex way of
reaching the SAME answer GS3D_Sampled already reaches via decomposition
tracking's much simpler independent-sample-then-minimum trick. Condor et
al. don't use that trick -- hence needing explicit segment tracking and a
bisection solver where GS3D_Sampled needs neither. Implement this because
you want a literature-faithful comparison baseline, not because it's more
correct than what you already have.

RANDOMNESS STRUCTURE IS DIFFERENT FROM EVERY OTHER FILE HERE: this draws
ONE random number up front (the overall interaction threshold along the
whole ray) and, if the hit lands in a multi-primitive segment, ONE more (to
choose which primitive gets credit, weighted by local density). It does NOT
draw one random number per candidate the way GS3D / GS3D_Sampled do.

A NEW KIND OF BOUND, STATED HONESTLY: ACTIVE_SET_CAP below bounds how many
primitives can be simultaneously overlapping AT ANY SINGLE POINT along the
ray -- not, like the old MAX_HITS, how many total hits exist anywhere along
the entire ray. These are genuinely different quantities: a ray can pass
through hundreds of Gaussians in total while never having more than a
handful truly overlapping at once. But it is still a cap, and if you build
a deliberately pathological scene with many primitives sharing nearly
identical centers, it can still be hit. Worth checking for in practice, not
worth pretending doesn't exist.
*/

// Abramowitz & Stegun 7.1.26 -- validated against scipy.special.erf,
// max abs error 1.4e-7 across [-6, 6].
float erf_approx(float x) {
    float s = sign(x);
    float ax = abs(x);
    float p = 0.3275911;
    float t = 1.0 / (1.0 + p * ax);
    float y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t
                     - 0.284496736)*t + 0.254829592) * t * exp(-ax*ax);
    return s * y;
}

// Giles' erfinv approximation (GPU Computing Gems 2010) -- the same one
// validated against scipy.special.erfinv (max abs error 2.4e-7) and
// already used in decomposition_tracking_GS_sampled.h.
float erfinv_approx(float x) {
    float w = -log((1.0 - x) * (1.0 + x));
    float p;
    if (w < 5.0) {
        w = w - 2.5;
        p = 2.81022636e-08;
        p = 3.43273939e-07 + p * w;
        p = -3.5233877e-06 + p * w;
        p = -4.39150654e-06 + p * w;
        p = 0.00021858087 + p * w;
        p = -0.00125372503 + p * w;
        p = -0.00417768164 + p * w;
        p = 0.246640727 + p * w;
        p = 1.50140941 + p * w;
    } else {
        w = sqrt(w) - 3.0;
        p = -0.000200214257;
        p = 0.000100950558 + p * w;
        p = 0.00134934322 + p * w;
        p = -0.00367342844 + p * w;
        p = 0.00573950773 + p * w;
        p = -0.0076224613 + p * w;
        p = 0.00943887047 + p * w;
        p = 1.00167406 + p * w;
        p = 2.83297682 + p * w;
    }
    return p * x;
}

float probit(float u) {
    return sqrt(2.0) * erfinv_approx(2.0 * u - 1.0);
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

    uint b1 = floatBitsToUint(w.x);
    uint b2 = floatBitsToUint(w.y);
    uint b3 = floatBitsToUint(w.z);
    uint seed = b1 ^ (b2 * 1973u) ^ (b3 * 9277u);
    rdv_rng_state = uvec4(seed, seed * 1664525u, ~seed, seed ^ 0x23F1u);
    random_step(); random_step();

    float sh_coefs[16];
    eval_sh(w, sh_coefs);

    const float K_SIGMA = 3.0;         // primitive support radius, matches
                                        // the 3-sigma AABB convention used
                                        // everywhere else in this project
    const int ACTIVE_SET_CAP = 64;
    const int SAFETY_MAX_ITERS = 4096; // defensive bound, not a quality cap

    int   active_idx[ACTIVE_SET_CAP];
    float active_exit_t[ACTIVE_SET_CAP];
    int   active_count = 0;

    float xi = random();               // ONE draw for the whole trace -- see header
    float T_cum = 1.0;
    float t_cursor = 0.0;
    vec3 final_color = vec3(0.0);
    bool resolved = false;

    for (int iter = 0; iter < SAFETY_MAX_ITERS && !resolved; ++iter) {

        // --- find the next NEW entry beyond t_cursor, via one fresh ray query ---
        int   new_idx = -1;
        float new_entry_t = 1e30;
        float new_A = 0.0, new_B = 0.0, new_C = 0.0;

        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, accelerationStructureEXT(parameters.ads),
            gl_RayFlagsOpaqueEXT, 0xFF, x, t_cursor, w, 10000.0);
        while (rayQueryProceedEXT(rq)) {
            if (rayQueryGetIntersectionTypeEXT(rq, false) ==
                gl_RayQueryCandidateIntersectionAABBEXT) {
                int i = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);

                // skip if already in the active set (its entry already happened)
                bool already_active = false;
                for (int a = 0; a < active_count; ++a)
                    if (active_idx[a] == i) { already_active = true; break; }
                if (already_active) continue;

                int cov_idx = i * 6;
                vec3 d = x - positions.data[i];
                float M00=inv_covs.data[cov_idx+0], M01=inv_covs.data[cov_idx+1], M02=inv_covs.data[cov_idx+2];
                float M11=inv_covs.data[cov_idx+3], M12=inv_covs.data[cov_idx+4], M22=inv_covs.data[cov_idx+5];

                float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                        + 2.0*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                        + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                        + M12*(w.y*d.z+w.z*d.y);
                if (A <= 1e-6) continue;
                float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z
                        + 2.0*(M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);

                // entry/exit of the K_SIGMA-radius bound: A t^2 + 2B t + (C - k^2) = 0
                float disc = B*B - A*(C - K_SIGMA*K_SIGMA);
                if (disc < 0.0) continue;  // ray misses this primitive's bound entirely
                float sq = sqrt(disc);
                float t_enter = (-B - sq) / A;
                float t_exit  = (-B + sq) / A;
                if (t_exit <= t_cursor) continue;      // fully behind us already
                float this_entry = max(t_enter, t_cursor + 1e-5);
                if (this_entry < new_entry_t) {
                    new_entry_t = this_entry; new_idx = i;
                    new_A = A; new_B = B; new_C = C;
                }
            }
        }

        // --- next EXIT among currently active primitives ---
        float exit_t = 1e30; int exit_slot = -1;
        for (int a = 0; a < active_count; ++a) {
            if (active_exit_t[a] < exit_t) { exit_t = active_exit_t[a]; exit_slot = a; }
        }

        bool have_entry = (new_idx >= 0);
        bool have_exit  = (exit_slot >= 0);
        if (!have_entry && !have_exit) break;  // ray has exited all geometry

        float event_t = have_entry ? (have_exit ? min(new_entry_t, exit_t) : new_entry_t) : exit_t;

        // --- process the segment [t_cursor, event_t] with the CURRENT active set ---
        if (active_count > 0) {
            float seg_ratio = 1.0;
            for (int a = 0; a < active_count; ++a) {
                int pi = active_idx[a];
                int cov_idx = pi * 6;
                vec3 d = x - positions.data[pi];
                float M00=inv_covs.data[cov_idx+0], M01=inv_covs.data[cov_idx+1], M02=inv_covs.data[cov_idx+2];
                float M11=inv_covs.data[cov_idx+3], M12=inv_covs.data[cov_idx+4], M22=inv_covs.data[cov_idx+5];
                float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                        + 2.0*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                        + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                        + M12*(w.y*d.z+w.z*d.y);
                float t_star = -B / A;
                float target_alpha = min(opacities.data[pi], 0.999);
                float peak_tau = -log(1.0 - target_alpha);
                // T_i(t) = exp(-peak_tau * Phi((t - t_star) * sqrt(A))); ratio over the segment:
                float u_lo = (t_cursor - t_star) * sqrt(A);
                float u_hi = (event_t   - t_star) * sqrt(A);
                float Phi_lo = 0.5 * (1.0 + erf_approx(u_lo * 0.70710678));
                float Phi_hi = 0.5 * (1.0 + erf_approx(u_hi * 0.70710678));
                seg_ratio *= exp(-peak_tau * (Phi_hi - Phi_lo));
            }
            float T_after = T_cum * seg_ratio;

            if (xi >= T_after) {
                // --- interaction happens inside THIS segment ---
                float t_hit;
                if (active_count == 1) {
                    int pi = active_idx[0];
                    int cov_idx = pi * 6;
                    vec3 d = x - positions.data[pi];
                    float M00=inv_covs.data[cov_idx+0], M01=inv_covs.data[cov_idx+1], M02=inv_covs.data[cov_idx+2];
                    float M11=inv_covs.data[cov_idx+3], M12=inv_covs.data[cov_idx+4], M22=inv_covs.data[cov_idx+5];
                    float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                            + 2.0*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                    float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                            + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                            + M12*(w.y*d.z+w.z*d.y);
                    float t_star = -B / A;
                    float target_alpha = min(opacities.data[pi], 0.999);
                    float peak_tau = -log(1.0 - target_alpha);
                    float target = xi / T_cum;
                    float uu = clamp(1.0 - log(target) / (-peak_tau), 1e-6, 1.0 - 1e-6);
                    t_hit = t_star + probit(uu) / sqrt(A);
                } else {
                    // more than one active primitive here: "revert to the
                    // bisection solver" -- matches the paper's own approach
                    float lo = t_cursor, hi = event_t;
                    for (int b = 0; b < 30; ++b) {
                        float mid = 0.5 * (lo + hi);
                        float ratio = 1.0;
                        for (int a = 0; a < active_count; ++a) {
                            int pi = active_idx[a];
                            int cov_idx = pi * 6;
                            vec3 d = x - positions.data[pi];
                            float M00=inv_covs.data[cov_idx+0], M01=inv_covs.data[cov_idx+1], M02=inv_covs.data[cov_idx+2];
                            float M11=inv_covs.data[cov_idx+3], M12=inv_covs.data[cov_idx+4], M22=inv_covs.data[cov_idx+5];
                            float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                                    + 2.0*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                            float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                                    + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                                    + M12*(w.y*d.z+w.z*d.y);
                            float t_star = -B / A;
                            float target_alpha = min(opacities.data[pi], 0.999);
                            float peak_tau = -log(1.0 - target_alpha);
                            float u_lo2 = (t_cursor - t_star) * sqrt(A);
                            float u_m   = (mid       - t_star) * sqrt(A);
                            float Phi_lo2 = 0.5 * (1.0 + erf_approx(u_lo2 * 0.70710678));
                            float Phi_m   = 0.5 * (1.0 + erf_approx(u_m   * 0.70710678));
                            ratio *= exp(-peak_tau * (Phi_m - Phi_lo2));
                        }
                        if (T_cum * ratio > xi) lo = mid; else hi = mid;
                    }
                    t_hit = 0.5 * (lo + hi);
                }

                // which primitive gets credit: weighted by LOCAL density at t_hit
                float local_w[ACTIVE_SET_CAP];
                float total_w = 0.0;
                for (int a = 0; a < active_count; ++a) {
                    int pi = active_idx[a];
                    int cov_idx = pi * 6;
                    vec3 d = x - positions.data[pi];
                    float M00=inv_covs.data[cov_idx+0], M01=inv_covs.data[cov_idx+1], M02=inv_covs.data[cov_idx+2];
                    float M11=inv_covs.data[cov_idx+3], M12=inv_covs.data[cov_idx+4], M22=inv_covs.data[cov_idx+5];
                    float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                            + 2.0*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                    float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                            + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                            + M12*(w.y*d.z+w.z*d.y);
                    float t_star = -B / A;
                    float target_alpha = min(opacities.data[pi], 0.999);
                    float peak_tau = -log(1.0 - target_alpha);
                    float sigma_peak = peak_tau * sqrt(A / 6.283185307);
                    local_w[a] = sigma_peak * exp(-0.5 * A * (t_hit - t_star) * (t_hit - t_star));
                    total_w += local_w[a];
                }
                float r = random() * max(total_w, 1e-12);
                float acc = 0.0;
                int winner = active_idx[0];
                for (int a = 0; a < active_count; ++a) {
                    acc += local_w[a];
                    if (r <= acc) { winner = active_idx[a]; break; }
                }

                vec3 gaussian_color = colors.data[winner] * sh_coefs[0];
                int rest_idx = winner * 45;
                for (int c = 1; c < 16; ++c) {
                    gaussian_color.x += f_rest.data[rest_idx + (c-1)     ] * sh_coefs[c];
                    gaussian_color.y += f_rest.data[rest_idx + (c-1) + 15] * sh_coefs[c];
                    gaussian_color.z += f_rest.data[rest_idx + (c-1) + 30] * sh_coefs[c];
                }
                final_color = clamp(gaussian_color + 0.5, 0.0, 1.0);
                resolved = true;
                break;
            }
            T_cum = T_after;
        }

        // --- ray survived this segment: advance and apply the event ---
        t_cursor = event_t;
        if (have_exit && (!have_entry || exit_t <= new_entry_t)) {
            // remove the exiting primitive (swap-with-last)
            active_idx[exit_slot]     = active_idx[active_count - 1];
            active_exit_t[exit_slot]  = active_exit_t[active_count - 1];
            active_count--;
        } else {
            // add the newly-entered primitive, if there's room
            if (active_count < ACTIVE_SET_CAP) {
                float disc = new_B*new_B - new_A*(new_C - K_SIGMA*K_SIGMA);
                float t_exit_new = (-new_B + sqrt(max(disc, 0.0))) / new_A;
                active_idx[active_count] = new_idx;
                active_exit_t[active_count] = t_exit_new;
                active_count++;
            }
            // if the active set is already full, this primitive is silently
            // dropped from consideration -- the ACTIVE_SET_CAP bound stated
            // in the header. Increase the cap if you hit this in practice.
        }
    }

    _output = float[](final_color.x, final_color.y, final_color.z);
}
BACKWARD {
    // Differentiation logic goes here
}
