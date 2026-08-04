/*
Decomposition Tracking for 3D Gaussian Splatting -- PROPERLY SAMPLED variant
Based on: "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"
          (Kutz, Habel, Li, Novak 2017)

Differs from decomposition_tracking_GS.h in exactly one conceptual way: the
original evaluates every candidate at its own fixed analytic peak (t* = -B/A)
and only randomizes WHETHER it is accepted. This version also samples WHERE
within the Gaussian's own support the interaction lands, via closed-form
inverse-CDF sampling of that Gaussian's 1D density profile along the ray.

WHY THIS MATTERS: two overlapping Gaussians A (nearer peak) and B (farther
peak) always resolve in peak order under the original file -- A's fixed peak
is always closer, so A wins every single trial whenever both are accepted,
deterministically. But a genuine random draw from B's density can land closer
to the camera than a genuine draw from A's density, in the region where their
tails overlap, and it SHOULD sometimes win there, proportional to how much of
each Gaussian's actual mass falls in that region. The original file can never
produce that outcome, because "where did Gaussian i interact" was never a
random variable -- only "did it interact at all" was.

This is decomposition tracking's usual construction, done correctly: sample
each candidate medium's own free-flight distance independently (not just an
accept/reject at one fixed point), then let the closest SAMPLED distance win.
The original file already got the "no explicit sort" part right; this fixes
the "sample a location, not just flip a coin at a fixed one" part.

Also fixed as a side effect: the original file used t_proj = dot(d_center, w)
(a cheap dot-product projection) both as a pre-filter and as the value stored
in closest_t. That only equals the true analytic peak -B/A for isotropic
Gaussians -- for anisotropic ("needle"/"pancake") ones, which real trained
3DGS scenes have plenty of, they can diverge enough that a genuinely closer
candidate gets pre-filtered out incorrectly. This version computes A, B, C
unconditionally (matching DontSplashYourGaussians.h's approach) and uses the
true peak throughout, trading that specific micro-optimization for
correctness.

VALIDATED NUMERICALLY before being ported here -- see overlap_ground_truth.py.
On a representative two-Gaussian overlap case, the original peak-only
construction was off by ~10-18% relative to the true joint integral; this
sampled version matched true ground truth to within Monte Carlo noise at a
couple million trials, and agreed with the peak-only version almost exactly
once the Gaussians no longer overlapped (confirming the gap is specifically
an overlap effect). The erfinv approximation below (Giles, GPU Computing
Gems 2010) matched scipy.special.erfinv to within 2.4e-7 across the full
domain in that same test, including deep in the near-opaque tail.

WHAT THIS DOES NOT FIX: decomposition_tracking_GS_ratio.h still resolves
overlap using each Gaussian's fixed peak for its depth-ordering comparisons.
Ratio tracking's whole design point is a DETERMINISTIC weighting rather than
a sampled location, so the trick below -- which relies on a genuine per-trial
random draw -- does not carry over to it directly. A correct ratio-tracking
analog would need to evaluate the COMBINED local density at shared points
along the ray, which is closer to Condor et al.'s interval-based integration
than to anything in either shader here. Treat that as a separate, still-open
problem rather than assuming this same fix applies there too.
*/

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

    random_step(); random_step();
    rdv_rng_state = random_seed();

    const float K_SIGMA = 3.0;   // matches their default ellipsoid shell extent

    float sh_coefs[16];
    eval_sh(w, sh_coefs);

    float closest_t = 10000.0;
    vec3 final_color = vec3(0.0);

    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, accelerationStructureEXT(parameters.ads),
        gl_RayFlagsNoneEXT, 0xFF, x, 0.0, w, 10000.0);

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
            

            //The Math: $A = w^T \Sigma^{-1} w$
            //What it does: It multiplies the ray's direction vector (w) by the shape matrix (M). 
            //This tells you how sharply the Gaussian is squished specifically along the direction you are looking.
            float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                    + 2.0*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);

            //The Math: $B = w^T \Sigma^{-1} d
            //What it does: d is the distance vector from the camera to the Gaussian. w is the ray direction. 
            //This cross-multiplies them both through the shape matrix. It measures how the ray's path aligns with the offset of the Gaussian.

            float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                    + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                    + M12*(w.y*d.z+w.z*d.y);

            

            if (A > 1e-6) {
                // true analytic peak -- see header note on why this replaces
                // the original file's dot-product t_proj
                //The derivative of $A t^2 + 2B t + C$ is $2A t + 2B = 0$.
                //Solve for $t$, and you get exactly $t = -B / A$.
                float t_star = -B / A;
                

                //The Math: $C = d^T \Sigma^{-1} d$
                //What it does: This multiplies the camera-to-Gaussian distance (d) against itself through the shape matrix. 
                //It acts as the baseline starting distance (the Mahalanobis distance from the camera to the center of the Gaussian).
                float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z
                        + 2.0*(M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);

                float disc = B*B - A*(C - K_SIGMA*K_SIGMA);
                if (disc < 0.0) continue;       

                //What it does: The math (C - (B*B)/A) calculates how close the ray actually gets to the physical center of the 3D Gaussian at its closest approach.
                //power: Calculates the exponent for the density.
                //The if statement: If power < -15.0, the ray passed so far away from the center that the fog is microscopically thin, so it ignores it. 
                //(Notice this is much more generous than Sun's -4.0 cutoff!).
                float power = -0.5 * (C - (B*B)/A);
                power = min(power, 0.0);
                if (power > -15.0) {

                    //This is the core physics math. A standard 3DGS rasterizer just treats opacity (target_alpha) like a piece of tinted glass. 
                    //But true volumetric ray tracing treats it like a cloud of particles.
                    //peak_tau: It takes the raw opacity and uses -log() to convert it into Optical Depth ($\tau$). 
                    //Optical depth is a physics term measuring how much physical "stuff" (particles) is actually floating in that space.
                    //exact_tau: Multiplies the total "stuff" by the exp(power) density dropoff. 
                        //This gives the exact amount of "stuff" the ray will pass through on its specific path.
                    //alpha: Converts that "stuff" back into a percentage probability (1.0 - exp(-exact_tau)).
                    float target_alpha = min(opacities.data[i], 0.999);
                    float peak_tau  = -log(1.0 - target_alpha);
                    float exact_tau = peak_tau * exp(power);
                    float alpha     = 1.0 - exp(-exact_tau);

                    // --- 1. does this Gaussian interact at all? (same test as before) ---
                    if (random() <= alpha) {

                        // --- 2. WHERE does it interact? sample from its OWN density
                        //        instead of always using t_star. This is the fix. ---
                        //float u = clamp(random(), 1e-6, 1.0 - 1e-6);



                        //This is the most important part of this specific shader.
                        //Instead of forcing the photon to stop exactly at t_star (which looks ghostly and wrong, as you saw in the Sun et al. render), 
                        //it scatters the hit dynamically!

                        //u: Rolls a brand new random number.

                        //probit(u): This is an advanced statistical function (the inverse cumulative distribution function of a normal curve). 
                        //If u is 0.5, probit returns 0. If u is 0.1, it returns a negative number. If u is 0.9, it returns a positive number.

                        //t_sample: It takes the peak (t_star) and pushes the hit location slightly forward or backward based on the probit function and the width of the Gaussian (sqrt(A)).
                        //float t_sample = t_star + probit(u) / sqrt(A);
                        float t_sample = t_star + random_normal()/ sqrt(A);

                        if (t_sample > 0.0 && t_sample < rayQueryGetIntersectionTEXT(rq, true)) {
                            //closest_t = t_sample;

                            rayQueryGenerateIntersectionEXT(rq, t_sample);

                            vec3 gaussian_color = colors.data[i] * sh_coefs[0];
                            int rest_idx = i * 45;
                            for (int c = 1; c < 16; ++c) {
                                gaussian_color.x += f_rest.data[rest_idx + (c-1)     ] * sh_coefs[c];
                                gaussian_color.y += f_rest.data[rest_idx + (c-1) + 15] * sh_coefs[c];
                                gaussian_color.z += f_rest.data[rest_idx + (c-1) + 30] * sh_coefs[c];
                            }
                            final_color = clamp(gaussian_color + 0.5, 0.0, 1.0);
                        }
                    }
                }
            }
        }
    }

    _output = float[](final_color.x, final_color.y, final_color.z);
}
BACKWARD {
    // Differentiation logic goes here
}
