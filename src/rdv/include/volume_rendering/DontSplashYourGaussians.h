/*
This map receives a ads data-structure with AABB for each gaussian
parameters:
- ads: Data structure to query the index of the AABB
- positions: each gaussian's position
- cov: each gaussian's covariance
- scale: each gaussian's scale
- sh: each gaussian's spherical harmonic coefficients
*/
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

    // 3. SH EVALUATION
    float sh_coefs[16];
    eval_sh(w, sh_coefs); 
    
    // 4. BVH INITIALIZATION
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, accelerationStructureEXT(parameters.ads), gl_RayFlagsOpaqueEXT, 0xFF, x, 0.0, w, 10000.0); 

    // INCREASED TO 128 TO HANDLE DENSE FOLIAGE
    int hit_indices[128];
    float hit_depths[128];
    int hit_count = 0;

    // --- PHASE 1: COLLECT EXACTLY THE CLOSEST 128 HITS ---
    while(rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
            int i = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            
            vec3 d = x - positions.data[i]; 
            int cov_idx = i * 6;
            float M00 = inv_covs.data[cov_idx + 0];
            float M01 = inv_covs.data[cov_idx + 1];
            float M02 = inv_covs.data[cov_idx + 2];
            float M11 = inv_covs.data[cov_idx + 3];
            float M12 = inv_covs.data[cov_idx + 4];
            float M22 = inv_covs.data[cov_idx + 5];

            float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z + 2.0 * (M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
            float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z + M01*(w.x*d.y + w.y*d.x) + M02*(w.x*d.z + w.z*d.x) + M12*(w.y*d.z + w.z*d.y);
            
            if (A > 1e-6) {
                float t_peak = -B / A; 
                
                if (t_peak > 0.0) {
                    if (hit_count < 128) {
                        hit_indices[hit_count] = i;
                        hit_depths[hit_count] = t_peak;
                        hit_count++;
                    } else {
                        // ARRAY IS FULL: Find the furthest hit we've collected so far
                        int max_idx = 0;
                        float max_depth = hit_depths[0];
                        for (int j = 1; j < 128; ++j) {
                            if (hit_depths[j] > max_depth) {
                                max_depth = hit_depths[j];
                                max_idx = j;
                            }
                        }
                        // If this new hit is CLOSER than our furthest stored hit, replace it!
                        if (t_peak < max_depth) {
                            hit_indices[max_idx] = i;
                            hit_depths[max_idx] = t_peak;
                        }
                    }
                }
            }
        }
    }

    // --- PHASE 2: FRONT-TO-BACK SORTING ---
    for (int i = 0; i < hit_count - 1; ++i) {
        for (int j = 0; j < hit_count - i - 1; ++j) {
            if (hit_depths[j] > hit_depths[j + 1]) {
                float temp_d = hit_depths[j];
                hit_depths[j] = hit_depths[j + 1];
                hit_depths[j + 1] = temp_d;
                int temp_i = hit_indices[j];
                hit_indices[j] = hit_indices[j + 1];
                hit_indices[j + 1] = temp_i;
            }
        }
    }

    // --- PHASE 3: GAUSSIAN ANALYTICAL ALPHA BLENDING ---
    vec3 final_color = vec3(0.0);
    float T = 1.0; 

    for (int h = 0; h < hit_count; ++h) {
        if (T < 0.01) break; 

        int i = hit_indices[h];
        
        vec3 d = x - positions.data[i]; 
        int cov_idx = i * 6;
        float M00 = inv_covs.data[cov_idx + 0];
        float M01 = inv_covs.data[cov_idx + 1];
        float M02 = inv_covs.data[cov_idx + 2];
        float M11 = inv_covs.data[cov_idx + 3];
        float M12 = inv_covs.data[cov_idx + 4];
        float M22 = inv_covs.data[cov_idx + 5];

        float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z + 2.0 * (M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
        float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z + 2.0 * (M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);
        float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z + M01*(w.x*d.y + w.y*d.x) + M02*(w.x*d.z + w.z*d.x) + M12*(w.y*d.z + w.z*d.y);

        if (A > 1e-6) {
            float power = -0.5 * (C - ((B * B) / A));
            
            if (power > -15.0 && power <= 0.0) {
                
                // REVERTED TO MONTE CARLO DENSITY LOGIC
                // (Matches the exact format the AI was trained on)
                float target_alpha = min(opacities.data[i], 0.999);
                float peak_tau = -log(1.0 - target_alpha);
                float exact_tau = peak_tau * exp(power);
                float alpha = 1.0 - exp(-exact_tau);

                // --- COLOR & SH CALCULATION ---
                vec3 gaussian_color = colors.data[i] * sh_coefs[0]; 
                int rest_idx = i * 45;
                for(int c = 1; c < 16; ++c) {
                    gaussian_color.x += f_rest.data[rest_idx + (c - 1) +  0] * sh_coefs[c]; 
                    gaussian_color.y += f_rest.data[rest_idx + (c - 1) + 15] * sh_coefs[c]; 
                    gaussian_color.z += f_rest.data[rest_idx + (c - 1) + 30] * sh_coefs[c]; 
                }
                
                gaussian_color = clamp(gaussian_color + 0.5, 0.0, 1.0);

                // --- COMPOSITE FRONT-TO-BACK ---
                float weight = alpha * T;
                final_color += weight * gaussian_color;
                T *= (1.0 - alpha);
            }
        }
    }

    // Optional: Add a white/fog background if T is still > 0
    // final_color += T * vec3(1.0, 1.0, 1.0);

    // 7. OUTPUT THE RESULT
    _output = float[](final_color.x, final_color.y, final_color.z);
}
BACKWARD {
    // Differentiation logic goes here
}