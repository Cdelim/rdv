/*
Decomposition Tracking for 3D Gaussian Splatting -- RAW DENSITY variant

Takes sigma_t (the true peak 3D extinction coefficient, sigma_0 in the
derivation below) directly, instead of a 3DGS-style "opacity" that gets
converted via -log(1-opacity). This isn't just a renaming to match Condor
et al.'s .ply schema -- it fixes a real approximation present in every
other shader in this project.

THE ISSUE, confirmed numerically before writing this file: tau_full(ray) =
sigma_0 * sqrt(2*pi/A), where A = w^T M w depends on the ray direction for
any ANISOTROPIC Gaussian. Every other shader here instead computes a FIXED
peak_tau = -log(1-opacity) once and reuses it for every ray direction via
peak_tau * exp(power) -- which is only exact if A doesn't depend on
direction, i.e. only for isotropic Gaussians. For a 20:1 aspect-ratio
primitive (checked numerically, not asserted), tau_full differs by 20x
between a ray through its thin axis and a ray along its wide plane. Thin
structures -- bicycle spokes, foliage, hair -- are exactly where real
trained 3DGS scenes have their most anisotropic Gaussians, so this
approximation is not a corner case.

Why the OTHER shaders still use -log(1-opacity): a standard 3DGS .ply only
gives you "opacity", a 2D-projection-based quantity from the original
rasterizer's billboard model -- there is no way to recover the true,
direction-independent sigma_0 from it after the fact. This file exists
specifically because Condor et al.'s .ply gives you sigma_t directly, so
there's no need for that approximation here at all.

Everything else matches decomposition_tracking_GS_sampled.h: accept/reject
via the real alpha, then sample WHERE within the primitive's support the
interaction lands (not always the peak), using the same validated
erfinv/probit machinery.

NOTE ON COLOR: this file expects an "albedo" input, not an SH-encoded
emitted color. Feeding albedo through the emission-absorption color model
below (same clamp(color+0.5,0,1) convention as every other shader here) is
a visualization convenience, not a physically correct render -- albedo
governs how a primitive scatters INCOMING light, and this pipeline has no
light source or phase function. Treat renders from this file as a
structural sanity check of the geometry/density fit, not a quantitative
comparison to what the original paper's own (lit, scattering-aware)
renderer would produce for the same data.
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
    GPUPtr sigma_t_ptr = load_tensor(parameters.sigma_t);   // raw peak density, NOT opacity
    float_ptr sigma_t = float_ptr(sigma_t_ptr);
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

    float closest_t = 10000.0;
    vec3 final_color = vec3(0.0);

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
                float t_star = -B / A;

                float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z
                        + 2.0*(M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);
                float power = -0.5 * (C - (B*B)/A);

                if (power > -15.0 && power <= 0.0) {
                    // exact, ray-dependent tau_full -- the actual fix, see header
                    float exact_tau = sigma_t.data[i] * sqrt(6.283185307 / A) * exp(power);
                    float alpha = 1.0 - exp(-min(exact_tau, 30.0));

                    if (random() < alpha) {
                        float u = clamp(random(), 1e-6, 1.0 - 1e-6);
                        float t_sample = t_star + probit(u) / sqrt(A);

                        if (t_sample > 0.0 && t_sample < closest_t) {
                            closest_t = t_sample;

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