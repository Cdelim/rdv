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
uint pcg_hash(inout uint state) {
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    state = state * 747796405u + 2891336453u;
    return (word >> 22u) ^ word;
}

// Helper: Convert uint to float [0, 1)
float step_rand(inout uint seed) {
    return float(pcg_hash(seed)) / 4294967296.0;
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

    GPUPtr majorant_ptr = load_tensor(parameters.majorant_buffer);
    float_ptr majorants = float_ptr(majorant_ptr);

    GPUPtr minorant_ptr = load_tensor(parameters.minorant_buffer);
    float_ptr minorants = float_ptr(minorant_ptr);

    GPUPtr min_bounds_ptr = load_tensor(parameters.grid_min);
    vec3_ptr min_bounds = vec3_ptr(min_bounds_ptr);

    GPUPtr size_bounds_ptr = load_tensor(parameters.grid_size);
    vec3_ptr size_bounds = vec3_ptr(size_bounds_ptr);

    GPUPtr control_color_ptr = load_tensor(parameters.control_color_buffer);
    vec3_ptr control_colors = vec3_ptr(control_color_ptr);

    //RAY SETUP
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    
    uint seed = uint(abs(x.x * 1973.0) + abs(w.y * 9277.0) + abs(w.z * 26699.0));
    seed = pcg_hash(seed);

    float T = 1.0; 
    vec3 A = vec3(0.0); // Final accumulated color
    float current_t = 0.0;
    
    int GRID_RES = 64; 
    rayQueryEXT rayQuery;

    uint step_count = 0;

    // 4. MAIN TRACKING LOOP
    while (T > 0.01) {
        step_count++;
        if(step_count > 1000) break; // Safety break to prevent infinite loops
        if (current_t > 1000.0) break; // Maximum scene depth

        vec3 current_pos = x + current_t * w;
        
        //Fixme
        //Normalize to [0, 1] for voxel indexing
        vec3 norm_pos = (current_pos - min_bounds.data[0]) / size_bounds.data[0];
        
        // Check if ray is outside the volume bounds
        if (any(lessThan(norm_pos, vec3(0.0))) || any(greaterThan(norm_pos, vec3(1.0)))) {
            current_t += 0.2; 
            continue; 
        }

        int vox_x = clamp(int(norm_pos.x * float(GRID_RES)), 0, GRID_RES - 1);
        int vox_y = clamp(int(norm_pos.y * float(GRID_RES)), 0, GRID_RES - 1);
        int vox_z = clamp(int(norm_pos.z * float(GRID_RES)), 0, GRID_RES - 1);
        int flat_idx = vox_x * (GRID_RES * GRID_RES) + vox_y * GRID_RES + vox_z;

        float mu_bar = majorants.data[flat_idx]; // Majorant (Combined)
        float mu_c   = minorants.data[flat_idx]; // Control Component (Minorant)

        if (mu_bar < 0.001) {
            current_t += 0.05;
            continue; 
        }

        // Sample distance from control 
        float t_c = (mu_c > 0.0) ? -log(1.0 - step_rand(seed)) / mu_c : 1e10;
        
        // Sample distance from residual
        float t_r = -log(1.0 - step_rand(seed)) / (mu_bar - mu_c);

        // Decision
        if (t_c < t_r) {
            current_t += t_c;
            continue;
            //fixme 
            //A = control_colors.data[flat_idx];
            //T = 0.0; 
            //break;
        } else {
            // Residual Collision: Must evaluate specific Gaussians
            current_t += t_r;
            vec3 residual_pos = x + current_t * w;

            // EVALUATE REAL DENSITY (Querying the ADS for nearby Gaussians)
            rayQueryInitializeEXT(rayQuery, accelerationStructureEXT(parameters.ads), 
                                  gl_RayFlagsOpaqueEXT, 0xFF, residual_pos, 0.0, w, 0.001); 

            float mu_t_real = 0.0;
            vec3 real_color = vec3(0.0);

            while(rayQueryProceedEXT(rayQuery)) {
                if (rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
                    int index = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);
                    vec3 d = residual_pos - positions.data[index];
                    
                    int cov_idx = index * 6;
                    float power = -0.5 * (
                        inv_covs.data[cov_idx + 0] * d.x * d.x + 
                        inv_covs.data[cov_idx + 3] * d.y * d.y + 
                        inv_covs.data[cov_idx + 5] * d.z * d.z +
                        2.0 * inv_covs.data[cov_idx + 1] * d.x * d.y + 
                        2.0 * inv_covs.data[cov_idx + 2] * d.x * d.z + 
                        2.0 * inv_covs.data[cov_idx + 4] * d.y * d.z
                    );

                    if (power > -12.0) {
                        float density = opacities.data[index] * exp(power);
                        mu_t_real += density;
                        real_color += colors.data[index] * density;
                    }
                }
            }

            // Probability of Residual Collision
            // prob = (Actual Density - Base) / (Max Possible Density - Base)
            float prob = clamp((mu_t_real - mu_c) / (mu_bar - mu_c), 0.0, 1.0);

            if (step_rand(seed) < prob) {
                // SUCCESS
                A = real_color / max(mu_t_real, 0.0001); 
                T = 0.0; 
                break;
            }
            //Null collision
        }
    }

    _output = float[](A.x, A.y, A.z);
}

BACKWARD {
    // Differentiation logic goes here
}