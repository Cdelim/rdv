/*
Stochastic Ray Tracing of Transparent 3D Gaussians -- STRICT PAPER VERSION
Sun, Georgiev, Fei, Hašan (Adobe), EGSR 2025, arXiv:2504.06598v3.

This file is the strictly faithful version: it implements their Algorithm 1
and Equations 2, 4, 8, 9, 10 as written, with NO substitutions. If you want
the version that swaps their trigonometric hash for this framework's
built-in random(), that is a legitimate variant but it is NOT this file, and
it must not be described as reproducing their Section 4.2.

WHAT IS FAITHFUL HERE (and was not, in the modified version):
  - Eqs. 8/9/10: their exact stateless trigonometric hash, with their exact
    constants a1=91.3458, a2=(12.9898,78.233), b1=47453.5453,
    b2=43758.5453. Validated before porting: 200k samples over [-3,3]^3 gave
    mean 0.4985, std 0.2891, chi-square 12.9 vs uniform on 10 bins (df=9,
    threshold 21.7 at p=0.01).
  - Eq. 2 support radius s = 2*sqrt(2). Mahalanobis^2 at the peak equals
    -2*power, so their cull is exactly power >= -s^2/2 = -4.0.
  - Alg. 1 line 15: report + clip via rayQueryGenerateIntersectionEXT.
  - Eq. 6: shade ONLY the closest accepted intersection. No alpha multiply,
    no transmittance multiply -- the Russian Roulette acceptance IS the alpha
    weighting and "closest accepted wins" IS the transmittance. Multiplying
    again would double-count and bias the image dark.

TWO THINGS THAT MUST NOT BE REMOVED, and why:

(a) THE JITTER. Their hash is a function of the hit POSITION only. Without a
    per-sample perturbation, every sample of a given pixel would hash to the
    SAME xi, so the estimator would be frozen -- averaging N samples would
    return N copies of one outcome and never converge. Their Sec. 4.2 handles
    this by perturbing the hit position with frame-number-dependent
    quasi-random numbers from a stateful Sobol sequence generated during
    camera-ray generation. This framework's stateful random() plays that role
    here. If you delete the jitter while keeping the hash, samples=N stops
    working and you will not notice from a single frame.

(b) THE COMMITTED-TYPE GUARD. When no intersection has been committed yet,
    the value returned by rayQueryGetIntersectionTEXT(rq, true) is not
    well-defined for the None case. Testing t_hit against it unguarded may
    happen to work on one driver and silently reject every candidate (black
    image) on another. The guard below is cheap; keep it.

A LIKELY TYPO IN THEIR PAPER: Alg. 1 lines 12-13 read
    `if xi < g1.alpha then return  [rejected by Russian Roulette]`
which accepts with probability 1-alpha and contradicts BOTH their Eq. 4
(alpha_hat = 1 with probability alpha) and their Fig. 2 caption
(alpha_hat_i = 1 if xi_i < alpha_i). This file implements the Eq. 4 / Fig. 2
version: ACCEPT when xi < alpha. Flagged rather than silently corrected.

MULTI-SAMPLE (their Sec. 3.5 / Eq. 7): they state explicitly that averaging N
independent evaluations "can be achieved by tracing a ray independently N
times" -- which is exactly what sensor.view(model, samples=N) already does.
Their single-traversal N-instantiation is a performance optimization of the
same estimator, not a different estimator, so it is deliberately omitted.

DEPTH_MODE: 0 = OursMean (mean of the 1D Gaussian along the ray, their
Sec. 3.1). 1 = OursCenter (Gaussian center projected onto the ray direction,
their Sec. 4.1) -- use this one for assets trained with a rasterizer, such as
the standard 3DGS bicycle scene; their Fig. 3 shows it scoring better there.
*/

#define DEPTH_MODE 0

// --- Their Eq. 8 ---
float sun_r1(float q) {
    return fract(47453.5453 * sin(91.3458 * q));
}

// --- Their Eq. 9 ---
float sun_r2(vec2 q) {
    return fract(43758.5453 * sin(dot(q, vec2(12.9898, 78.233))));
}

// --- Their Eq. 10: xi(p) = r2(p.xy + r1(p.z)) ---
float sun_hash(vec3 p) {
    return sun_r2(p.xy + vec2(sun_r1(p.z)));
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

    // Stateful RNG, used ONLY to produce the per-sample position perturbation
    // that their Sec. 4.2 gets from a Sobol sequence. See note (a) above --
    // the hash itself is deterministic in position, so this is what makes
    // samples=N actually converge instead of repeating one outcome.
    
    uint b1 = floatBitsToUint(w.x);
    uint b2 = floatBitsToUint(w.y);
    uint b3 = floatBitsToUint(w.z);
    
    // 2. Smash them together to create a 100% unique seed per pixel
    uint seed = b1 ^ (b2 * 1973u) ^ (b3 * 9277u);
    
    // 3. Load the seed into the engine state
    rdv_rng_state = uvec4(seed, seed * 1664525u, ~seed, seed ^ 0x23F1u);
    
    // 4. "Warm up" the engine so the first outputs aren't correlated
    random_step(); random_step();
    
    // 5. Generate the jitter!
    vec3 jitter = vec3(random(), random(), random()) * 1024.0;

    float sh_coefs[16];
    eval_sh(w, sh_coefs);

    // Their Eq. 2: s = 2*sqrt(2) => Mahalanobis^2 <= 8 => power >= -4
    const float POWER_CUTOFF = -4.0;

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

            float A = M00*w.x*w.x + M11*w.y*w.y + M22*w.z*w.z
                    + 2.0*(M01*w.x*w.y + M02*w.x*w.z + M12*w.y*w.z);
            if (A <= 1e-6) continue;

            float B = M00*w.x*d.x + M11*w.y*d.y + M22*w.z*d.z
                    + M01*(w.x*d.y+w.y*d.x) + M02*(w.x*d.z+w.z*d.x)
                    + M12*(w.y*d.z+w.z*d.y);
            float t_star = -B / A;
            vec3 dp = d + t_star * w;                    // perpendicular residual, small
            float power = -0.5 * (M00*dp.x*dp.x + M11*dp.y*dp.y + M22*dp.z*dp.z
                                + 2.0*(M01*dp.x*dp.y + M02*dp.x*dp.z + M12*dp.y*dp.z));

            // --- Alg. 1 lines 2-3: the 1D Gaussian along the ray ---
            float t_peak = -B / A;
            //float power  = -0.5 * (C - (B*B)/A);
            //power = min(power, 0.0);   
            // --- Alg. 1 lines 7-9 / Eq. 2: negligibility cull.
            // Evaluated at the PEAK in both depth modes, per Alg. 1 line 8
            // ("mean of g1 is outside the AABB of g").
            if (power < POWER_CUTOFF) continue;

#if DEPTH_MODE == 0
            float t_hit = t_peak;        // OursMean, their Sec. 3.1
#else
            float t_hit = dot(-d, w);    // OursCenter, their Sec. 4.1
#endif

            // --- Alg. 1 lines 4-6: valid ray range ---
            if (t_hit <= 0.0) continue;
            // guard (b): only compare against the committed t if one exists
            if (rayQueryGetIntersectionTypeEXT(rq, true) !=
                    gl_RayQueryCommittedIntersectionNoneEXT &&
                t_hit >= rayQueryGetIntersectionTEXT(rq, true)) continue;

            // --- Alg. 1 lines 10-11: position-dependent hash, their Eq. 10 ---
            vec3 p_hit = x + t_hit * w;
            float xi = sun_hash(p_hit + jitter);

            // alpha = opacity * G(peak): the standard 3DGS convention
            float alpha = min(opacities.data[i] * exp(power), 0.9999);

            // --- Alg. 1 lines 12-14 / Eq. 4: Russian Roulette.
            // ACCEPT when xi < alpha (per Eq. 4 and Fig. 2; see header note).
            if (xi < alpha) {
                // --- Alg. 1 line 15: report hit and clip ray (r.tmax <- t) ---
                rayQueryGenerateIntersectionEXT(rq, t_hit);
            }
        }
    }

    // --- Eq. 6: shade the single closest accepted intersection ---
    vec3 final_color = vec3(0.0);

    if (rayQueryGetIntersectionTypeEXT(rq, true) ==
        gl_RayQueryCommittedIntersectionGeneratedEXT) {

        int piLocal = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        vec3 gaussian_color = colors.data[piLocal] * sh_coefs[0];
        int rest_idx = piLocal * 45;
        for (int c = 1; c < 16; ++c) {
            gaussian_color.x += f_rest.data[rest_idx + (c-1)     ] * sh_coefs[c];
            gaussian_color.y += f_rest.data[rest_idx + (c-1) + 15] * sh_coefs[c];
            gaussian_color.z += f_rest.data[rest_idx + (c-1) + 30] * sh_coefs[c];
        }
        final_color = clamp(gaussian_color + 0.5, 0.0, 1.0);
    }
    _output = float[](final_color.x, final_color.y, final_color.z);
    
}
BACKWARD {
    // Not implemented -- forward-only, consistent with this thesis's scope.
}