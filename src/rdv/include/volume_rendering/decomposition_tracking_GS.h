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

   // 1. Set up the Ray
   vec3 x = vec3(_input[0], _input[1], _input[2]);
   vec3 w = normalize(vec3(_input[3], _input[4], _input[5]));

   // 2. Initialize the Framework's RNG State!
   // Create a unique seed based on the ray's origin and direction
   // Use raw binary float bits to guarantee a unique seed per pixel!
    uint b1 = floatBitsToUint(w.x);
    uint b2 = floatBitsToUint(w.y);
    uint b3 = floatBitsToUint(w.z);
    uint seed = b1 ^ (b2 * 1973u) ^ (b3 * 9277u);

    // Mix it with a changing frame number to get different noise every frame
    rdv_rng_state = uvec4(seed, seed * 1664525u, ~seed, seed ^ 0x23F1u);
    random_step(); random_step(); // Double warm-up for safety

    // 3. STORAGE FOR SORTING (Max 64 overlaps per ray)
    const int MAX_HITS = 256;
    int hit_indices[MAX_HITS];
    float hit_distances[MAX_HITS];
    int hit_count = 0;

    // 4. QUERY THE HARDWARE BVH
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, accelerationStructureEXT(parameters.ads), gl_RayFlagsOpaqueEXT, 0xFF, x, 0.0, w, 1000.0); 

    while(rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
            int index = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            
            // Calculate approximate distance along the ray to the Gaussian's center
            vec3 d_center = positions.data[index] - x;
            float t_proj = dot(d_center, w); 

            // Only store Gaussians that are IN FRONT of the camera, up to the max limit
            if (t_proj > 0.0 && hit_count < MAX_HITS) {
                hit_indices[hit_count] = index;
                hit_distances[hit_count] = t_proj;
                hit_count++;
            }
        }
    }

    // 5. SORT THE HITS FRONT-TO-BACK (Insertion Sort inside the GPU!)
    for (int i = 1; i < hit_count; ++i) {
        int key_index = hit_indices[i];
        float key_dist = hit_distances[i];
        int j = i - 1;

        while (j >= 0 && hit_distances[j] > key_dist) {
            hit_indices[j + 1] = hit_indices[j];
            hit_distances[j + 1] = hit_distances[j];
            j = j - 1;
        }
        hit_indices[j + 1] = key_index;
        hit_distances[j + 1] = key_dist;
    }

    // 6. THE PROFESSOR'S ANALYTICAL STOCHASTIC EVALUATION
    vec3 final_color = vec3(0.0); // Default background is White

    for (int k = 0; k < hit_count; ++k) {
        int i = hit_indices[k];
        int cov_idx = i * 6;
        
        vec3 d = x - positions.data[i]; 

        // Load the Inverse Covariance Matrix
        float M00 = inv_covs.data[cov_idx + 0];
        float M01 = inv_covs.data[cov_idx + 1];
        float M02 = inv_covs.data[cov_idx + 2];
        float M11 = inv_covs.data[cov_idx + 3];
        float M12 = inv_covs.data[cov_idx + 4];
        float M22 = inv_covs.data[cov_idx + 5];

        // Analytical Quadratic Variables
        float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z + 
                  2.0 * (M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);

        float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z + 
                  2.0 * (M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);

        float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z + 
                  M01*(w.x*d.y + w.y*d.x) + M02*(w.x*d.z + w.z*d.x) + M12*(w.y*d.z + w.z*d.y);

        if (A > 1e-6) {
            float power = -0.5 * (C - ((B * B) / A));

            // If the math isn't astronomical, evaluate the exact area under the curve
            if (power > -15.0 && power <= 0.0) {
               // 1. Convert the 2D PLY opacity into a peak 3D Optical Depth
                // We clamp it to 0.999 so the log() function doesn't hit infinity!
                float target_alpha = min(opacities.data[i], 0.999);
                float peak_tau = -log(1.0 - target_alpha);

                // 2. Apply the true 3D Volumetric bell-curve falloff along the ray
                float exact_tau = peak_tau * exp(power);
                
                // 3. Convert back to a probability for the dice roll
                float hit_probability = 1.0 - exp(-exact_tau);

                // ROLL THE DICE!
                if (step_rand(seed) < hit_probability) {
                    final_color = colors.data[i];
                    break; 
                }
                // If we missed, the loop continues to the next closest Gaussian...
            }
        }
    }

    // 7. OUTPUT THE RESULT
    _output = float[](final_color.x, final_color.y, final_color.z);
}

BACKWARD {
    // Differentiation logic goes here
}