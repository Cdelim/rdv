/*
Don't Splat Your Gaussians -- FAITHFUL implementation (v3)
*/

// Abramowitz & Stegun 7.1.26
float erf_approx(float x) {
    float s = sign(x);
    float ax = abs(x);
    float p = 0.3275911f;
    float t = 1.0f / (1.0f + p * ax);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                     - 0.284496736f) * t + 0.254829592f) * t * exp(-ax * ax);
    return s * y;
}

// Giles' erfinv approximation (GPU Computing Gems 2010)
float erfinv_approx(float x) {
    float w = -log((1.0f - x) * (1.0f + x));
    float p;
    if (w < 5.0f) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

float probit(float u) {
    return sqrt(2.0f) * erfinv_approx(2.0f * u - 1.0f);
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

    const float K_SIGMA = 3.0f;         
    const int ACTIVE_SET_CAP = 64;
    const int SAFETY_MAX_ITERS = 4096; 

    int   active_idx[64];
    float active_exit_t[64];
    int   active_count = 0;

    float xi = random();               
    float T_cum = 1.0f;
    float t_cursor = 0.0f;
    vec3 final_color = vec3(0.0f);
    bool resolved = false;

    for (int iter = 0; iter < SAFETY_MAX_ITERS && !resolved; ++iter) {

        int   new_idx = -1;
        float new_entry_t = 1.0e30f;
        float new_A = 0.0f;
        float new_B = 0.0f; 
        float new_C = 0.0f;

        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, accelerationStructureEXT(parameters.ads),
            gl_RayFlagsOpaqueEXT, 0xFF, x, t_cursor, w, 10000.0f);
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
                        + 2.0f*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                        + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                        + M12*(w.y*d.z+w.z*d.y);
                if (A <= 1.0e-6f) continue;
                float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z
                        + 2.0f*(M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);

                float disc = B*B - A*(C - K_SIGMA*K_SIGMA);
                if (disc < 0.0f) continue;  
                float sq = sqrt(disc);
                float t_enter = (-B - sq) / A;
                float t_exit  = (-B + sq) / A;
                if (t_exit <= t_cursor) continue;      
                float this_entry = max(t_enter, t_cursor + 1.0e-5f);
                if (this_entry < new_entry_t) {
                    new_entry_t = this_entry; new_idx = i;
                    new_A = A; new_B = B; new_C = C;
                }
            }
        }

        float exit_t = 1.0e30f; 
        int exit_slot = -1;
        for (int a = 0; a < active_count; ++a) {
            if (active_exit_t[a] < exit_t) { exit_t = active_exit_t[a]; exit_slot = a; }
        }

        bool have_entry = (new_idx >= 0);
        bool have_exit  = (exit_slot >= 0);
        if (!have_entry && !have_exit) break;  

        float event_t = have_entry ? (have_exit ? min(new_entry_t, exit_t) : new_entry_t) : exit_t;

        float T_after = T_cum;
        if (active_count > 0) {
            float seg_ratio = 1.0f;
            for (int a = 0; a < active_count; ++a) {
                int piLoc = active_idx[a];
                int cov_idx = piLoc * 6;
                vec3 d = x - positions.data[piLoc];
                float M00 = inv_covs.data[cov_idx+0];
                float M01 = inv_covs.data[cov_idx+1];
                float M02 = inv_covs.data[cov_idx+2];
                float M11 = inv_covs.data[cov_idx+3];
                float M12 = inv_covs.data[cov_idx+4];
                float M22 = inv_covs.data[cov_idx+5];
                float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                        + 2.0f*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                        + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                        + M12*(w.y*d.z+w.z*d.y);
                float t_star = -B / A;
                float target_alpha = min(opacities.data[piLoc], 0.999f);
                float peak_tau = -log(1.0f - target_alpha);
                float u_lo = (t_cursor - t_star) * sqrt(A);
                float u_hi = (event_t   - t_star) * sqrt(A);
                float Phi_lo = 0.5f * (1.0f + erf_approx(u_lo * 0.70710678f));
                float Phi_hi = 0.5f * (1.0f + erf_approx(u_hi * 0.70710678f));
                seg_ratio *= exp(-peak_tau * (Phi_hi - Phi_lo));
            }
            T_after = T_cum * seg_ratio;

            if (xi >= T_after) {
                float t_hit;
                if (active_count == 1) {
                    int piLoc = active_idx[0];
                    int cov_idx = piLoc * 6;
                    vec3 d = x - positions.data[piLoc];
                    float M00 = inv_covs.data[cov_idx+0];
                    float M01 = inv_covs.data[cov_idx+1];
                    float M02 = inv_covs.data[cov_idx+2];
                    float M11 = inv_covs.data[cov_idx+3];
                    float M12 = inv_covs.data[cov_idx+4];
                    float M22 = inv_covs.data[cov_idx+5];
                    float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                            + 2.0f*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                    float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                            + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                            + M12*(w.y*d.z+w.z*d.y);
                    float t_star = -B / A;
                    float target_alpha = min(opacities.data[piLoc], 0.999f);
                    float peak_tau = -log(1.0f - target_alpha);
                    float target = xi / T_cum;
                    float uu = clamp(1.0f - log(target) / (-peak_tau), 1.0e-6f, 0.999999f);
                    t_hit = t_star + probit(uu) / sqrt(A);
                } else {
                    float lo = t_cursor;  
                    float hi = event_t;
                    for (int b = 0; b < 30; ++b) {
                        float mid = 0.5f * (lo + hi);
                        float ratio = 1.0f;
                        for (int a = 0; a < active_count; ++a) {
                            int piLoc = active_idx[a];
                            int cov_idx = piLoc * 6;
                            vec3 d = x - positions.data[piLoc];
                            float M00 = inv_covs.data[cov_idx+0];
                            float M01 = inv_covs.data[cov_idx+1];
                            float M02 = inv_covs.data[cov_idx+2];
                            float M11 = inv_covs.data[cov_idx+3];
                            float M12 = inv_covs.data[cov_idx+4];
                            float M22 = inv_covs.data[cov_idx+5];
                            float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                                    + 2.0f*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                            float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                                    + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                                    + M12*(w.y*d.z+w.z*d.y);
                            float t_star = -B / A;
                            float target_alpha = min(opacities.data[piLoc], 0.999f);
                            float peak_tau = -log(1.0f - target_alpha);
                            float u_lo2 = (t_cursor - t_star) * sqrt(A);
                            float u_m   = (mid       - t_star) * sqrt(A);
                            float Phi_lo2 = 0.5f * (1.0f + erf_approx(u_lo2 * 0.70710678f));
                            float Phi_m   = 0.5f * (1.0f + erf_approx(u_m   * 0.70710678f));
                            ratio *= exp(-peak_tau * (Phi_m - Phi_lo2));
                        }
                        if (T_cum * ratio > xi) lo = mid; else hi = mid;
                    }
                    t_hit = 0.5f * (lo + hi);
                }

                float local_w[64];
                float total_w = 0.0f;
                for (int a = 0; a < active_count; ++a) {
                    int piLoc = active_idx[a];
                    int cov_idx = piLoc * 6;
                    vec3 d = x - positions.data[piLoc];
                    float M00 = inv_covs.data[cov_idx+0];
                    float M01 = inv_covs.data[cov_idx+1];
                    float M02 = inv_covs.data[cov_idx+2];
                    float M11 = inv_covs.data[cov_idx+3];
                    float M12 = inv_covs.data[cov_idx+4];
                    float M22 = inv_covs.data[cov_idx+5];
                    float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                            + 2.0f*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
                    float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                            + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                            + M12*(w.y*d.z+w.z*d.y);
                    float t_star = -B / A;
                    float target_alpha = min(opacities.data[piLoc], 0.999f);
                    float peak_tau = -log(1.0f - target_alpha);
                    float sigma_peak = peak_tau * sqrt(A / 6.283185307f);
                    local_w[a] = sigma_peak * exp(-0.5f * A * (t_hit - t_star) * (t_hit - t_star));
                    total_w += local_w[a];
                }
                float r = random() * max(total_w, 1.0e-12f);
                float acc = 0.0f;
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
                final_color = clamp(gaussian_color + 0.5f, 0.0f, 1.0f);
                resolved = true;
                break;
            }
        }

        T_cum = T_after;
        t_cursor = event_t;
        if (have_exit && (!have_entry || exit_t <= new_entry_t)) {
            active_idx[exit_slot]     = active_idx[active_count - 1];
            active_exit_t[exit_slot]  = active_exit_t[active_count - 1];
            active_count--;
        } else {
            if (active_count < ACTIVE_SET_CAP) {
                float disc = new_B*new_B - new_A*(new_C - K_SIGMA*K_SIGMA);
                float t_exit_new = (-new_B + sqrt(max(disc, 0.0f))) / new_A;
                active_idx[active_count] = new_idx;
                active_exit_t[active_count] = t_exit_new;
                active_count++;
            }
        }
    }

    _output = float[](final_color.x, final_color.y, final_color.z);
}

BACKWARD {
}