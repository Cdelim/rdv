/*
Don't Splat Your Gaussians -- VPRF integrator (Equation 22)

This is a DIFFERENT algorithm from DontSplashYourGaussians_v3_faithful.h,
not a fix to it -- both are faithful to the paper, to two different parts
of it. The paper implements two integrators: VPPT (stochastic, general
scattering media, what v3_faithful.h implements) and VPRF (deterministic,
radiance-field-only, Section 6.1: "In VPRF, process_segment() computes the
radiance field contribution of that segment following Equation (22)").
Section 7.3 -- the section that actually compares against 3DGS -- states
explicitly: "For this application, we use our simpler volumetric primitive
radiance field (VPRF) integrator." That's this file.

THE ALGORITHM (Equation 22): walk segments front-to-back exactly like
DontSplashYourGaussians_v2.h does, but where v2 attributes each segment's
transmittance drop entirely to ONE primitive (deterministic sorted alpha
blend, single primitive per event), this treats a segment's active set as a
GROUP: compute each active primitive's own optical depth tau_i integrated
across JUST this segment, sum them to tau_total, and the segment's overall
alpha is 1-exp(-tau_total); the segment's color is a tau_i/tau_total
weighted average of the active primitives' colors. No random numbers. No
bisection. Structurally this is the simplest file in the whole project --
it's DSYG_v2's exact accumulation loop, generalized from "one primitive per
segment" to "a weighted blend of however many are active."

VALIDATED IN PYTHON FIRST, and the result is the whole point of this
header comment: on the same two-Gaussian overlap scene used throughout
this project, Equation 22 gave [0.4445, 0, 0.3947] against a true ground
truth of [0.5454, 0, 0.2946] -- about 34% relative error, systematically
UNDER-crediting the nearer primitive and OVER-crediting the farther one.
This is not a porting bug; the same Python computation gives the same
answer. It's the approximation the paper names and accepts, tracing to a
real simplification: the tau_i/tau_total weighting treats a segment's
composition as uniform across its extent, ignoring that transmittance
should weight the near part of a wide segment more heavily than the far
part. DontSplashYourGaussians_v3_faithful.h does not make this
simplification and matched true ground truth to under 1% on the same scene.

USE THIS FILE WHEN: you want to compare against what Condor et al. actually
reported for radiance-field / 3DGS-style applications, warts and all.
USE v3_faithful.h WHEN: you want the most physically accurate baseline
achievable from the same primitive representation, independent of which
practical shortcut the original paper's authors chose for speed.
Reporting BOTH, and being explicit in your thesis about which question each
one answers, is more defensible than picking one and treating it as "the"
faithful baseline -- they're both faithful, to different things.
*/

// Abramowitz & Stegun 7.1.26 -- validated against scipy.special.erf,
// max abs error 1.4e-7 across [-6, 6]. Same function as in
// DontSplashYourGaussians_v3_faithful.h; this file needs ONLY this one --
// no erfinv_approx, no probit -- since Equation 22 never inverts anything:
// no sampling, no bisection, by design.
float erf_approx(float x) {
    float s = sign(x);
    float ax = abs(x);
    float p = 0.3275911;
    float t = 1.0 / (1.0 + p * ax);
    float y = 1.0 - (((((1.061405429*t - 1.453152027)*t) + 1.421413741)*t
                     - 0.284496736)*t + 0.254829592) * t * exp(-ax*ax);
    return s * y;
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

    float sh_coefs[16];
    eval_sh(w, sh_coefs);

    const float K_SIGMA = 3.0;
    const int ACTIVE_SET_CAP = 64;      // same bound, same honesty as v3_faithful.h:
                                          // caps simultaneous overlap at a point, not
                                          // total hits along the ray
    const int SAFETY_MAX_ITERS = 4096;

    int   active_idx[ACTIVE_SET_CAP];
    float active_exit_t[ACTIVE_SET_CAP];
    int   active_count = 0;

    float T = 1.0;
    float t_cursor = 0.0;
    vec3 final_color = vec3(0.0);

    for (int iter = 0; iter < SAFETY_MAX_ITERS; ++iter) {
        if (T < 0.001) break;

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

                bool already_active = false;
                for (int a = 0; a < active_count; ++a)
                    if (active_idx[a] == i) { already_active = true; break; }
                if (already_active) continue;

                int cov_idx = i * 6;
                vec3 d = x - positions.data[i];
                float M00 = inv_covs.data[cov_idx+0];
                float M01 = inv_covs.data[cov_idx+1];
                float M02 = inv_covs.data[cov_idx+2];
                float M11 = inv_covs.data[cov_idx+3];
                float M12 = inv_covs.data[cov_idx+4];
                float M22 = inv_covs.data[cov_idx+5];

                float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                        + 2.0*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                        + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                        + M12*(w.y*d.z+w.z*d.y);
                if (A <= 1e-6) continue;
                float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z
                        + 2.0*(M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);

                float disc = B*B - A*(C - K_SIGMA*K_SIGMA);
                if (disc < 0.0) continue;
                float sq = sqrt(disc);
                float t_enter = (-B - sq) / A;
                float t_exit  = (-B + sq) / A;
                if (t_exit <= t_cursor) continue;
                float this_entry = max(t_enter, t_cursor + 1e-5);
                if (this_entry < new_entry_t) {
                    new_entry_t = this_entry; new_idx = i;
                    new_A = A; new_B = B; new_C = C;
                }
            }
        }

        float exit_t = 1e30; int exit_slot = -1;
        for (int a = 0; a < active_count; ++a)
            if (active_exit_t[a] < exit_t) { exit_t = active_exit_t[a]; exit_slot = a; }

        bool have_entry = (new_idx >= 0);
        bool have_exit  = (exit_slot >= 0);
        if (!have_entry && !have_exit) break;

        float event_t = have_entry ? (have_exit ? min(new_entry_t, exit_t) : new_entry_t) : exit_t;

        // --- Equation 22: this segment's deterministic contribution ---
        if (active_count > 0) {
            float tau_total = 0.0;
            float tau_i_arr[ACTIVE_SET_CAP];

            for (int a = 0; a < active_count; ++a) {
                int piLocal = active_idx[a];
                int cov_idx = piLocal * 6;
                vec3 d = x - positions.data[piLocal];
                float M00 = inv_covs.data[cov_idx+0];
                float M01 = inv_covs.data[cov_idx+1];
                float M02 = inv_covs.data[cov_idx+2];
                float M11 = inv_covs.data[cov_idx+3];
                float M12 = inv_covs.data[cov_idx+4];
                float M22 = inv_covs.data[cov_idx+5];
                float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                        + 2.0*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                        + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                        + M12*(w.y*d.z+w.z*d.y);
                float t_star = -B / A;
                float target_alpha = min(opacities.data[piLocal], 0.999);
                float peak_tau = -log(1.0 - target_alpha);

                // tau_i integrated across JUST this segment [t_cursor, event_t]
                float u_lo = (t_cursor - t_star) * sqrt(A);
                float u_hi = (event_t   - t_star) * sqrt(A);
                float Phi_lo = 0.5 * (1.0 + erf_approx(u_lo * 0.70710678));
                float Phi_hi = 0.5 * (1.0 + erf_approx(u_hi * 0.70710678));
                float tau_i = peak_tau * (Phi_hi - Phi_lo);

                tau_i_arr[a] = tau_i;
                tau_total += tau_i;
            }

            if (tau_total > 1e-8) {
                float alpha_k = 1.0 - exp(-tau_total);
                vec3 seg_color = vec3(0.0);

                for (int a = 0; a < active_count; ++a) {
                    int piLocal = active_idx[a];
                    float weight = tau_i_arr[a] / tau_total;

                    vec3 gaussian_color = colors.data[piLocal] * sh_coefs[0];
                    int rest_idx = piLocal * 45;
                    for (int c = 1; c < 16; ++c) {
                        gaussian_color.x += f_rest.data[rest_idx + (c-1)     ] * sh_coefs[c];
                        gaussian_color.y += f_rest.data[rest_idx + (c-1) + 15] * sh_coefs[c];
                        gaussian_color.z += f_rest.data[rest_idx + (c-1) + 30] * sh_coefs[c];
                    }
                    gaussian_color = clamp(gaussian_color + 0.5, 0.0, 1.0);

                    seg_color += weight * gaussian_color;
                }

                final_color += T * alpha_k * seg_color;
                T *= exp(-tau_total);
            }
        }

        t_cursor = event_t;
        if (have_exit && (!have_entry || exit_t <= new_entry_t)) {
            active_idx[exit_slot]    = active_idx[active_count - 1];
            active_exit_t[exit_slot] = active_exit_t[active_count - 1];
            active_count--;
        } else {
            if (active_count < ACTIVE_SET_CAP) {
                float disc = new_B*new_B - new_A*(new_C - K_SIGMA*K_SIGMA);
                float t_exit_new = (-new_B + sqrt(max(disc, 0.0))) / new_A;
                active_idx[active_count] = new_idx;
                active_exit_t[active_count] = t_exit_new;
                active_count++;
            }
        }
    }

    _output = float[](final_color.x, final_color.y, final_color.z);
}
BACKWARD {
    // Differentiation logic goes here
}
