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
    eval_sh(-w, sh_coefs); 

    float closest_t = 10000.0;


    //  DECOMPOSITION TRACKING — Monte Carlo stochastic estimator
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


    vec3 final_color = vec3(0.0); // Black background
    
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, accelerationStructureEXT(parameters.ads), gl_RayFlagsOpaqueEXT, 0xFF, x, 0.0, w, 10000.0); 

    while(rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionAABBEXT) {
            int i = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            
            vec3 d_center = positions.data[i] - x;
            float t_proj = dot(d_center, w); 

            // MASSIVE OPTIMIZATION: 
            // Only do the heavy math if this Gaussian is in front of the camera 
            // AND closer than our current best hit!
            if (t_proj > 0.0 && t_proj < closest_t) {
                
                int cov_idx = i * 6;
                vec3 d = x - positions.data[i]; 

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
                        float target_alpha = min(opacities.data[i], 0.999);
                        float peak_tau = -log(1.0 - target_alpha);
                        float exact_tau = peak_tau * exp(power);
                        float alpha = 1.0 - exp(-exact_tau);

                        // --- THE MONTE CARLO DICE ROLL ---
                        if (random() < alpha) {
                            
                            // WE HIT IT! Update the closest boundary so we can ignore 
                            // anything further away for the rest of the BVH search!
                            closest_t = t_proj;
                            
                            vec3 gaussian_color = colors.data[i] * sh_coefs[0]; 

                            // 2. Add the 15 Rest Coefficients (Degrees 1, 2, and 3)
                            int rest_idx = i * 45;
                            for(int c = 1; c < 16; ++c) {
                                // The PLY file stores f_rest as: 15 Reds, 15 Greens, 15 Blues!
                                gaussian_color.x += f_rest.data[rest_idx + (c - 1) +  0] * sh_coefs[c]; // Red
                                gaussian_color.y += f_rest.data[rest_idx + (c - 1) + 15] * sh_coefs[c]; // Green
                                gaussian_color.z += f_rest.data[rest_idx + (c - 1) + 30] * sh_coefs[c]; // Blue
                            }

                            // 3. Center color space, clamp, and WE ARE DONE. 
                            final_color = clamp(gaussian_color + 0.5, 0.0, 1.0);
                        }
                    }
                }
            }
        }
    }

    // 7. OUTPUT THE RESULT
    _output = float[](final_color.x, final_color.y, final_color.z);
}
BACKWARD {
    // Differentiation logic goes here
}