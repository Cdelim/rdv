/*
Don't Splat Your Gaussians -- VPRF, LOCAL-SPACE FORMULATION (v6)

Same algorithm as v5 (faithful to volprim_rf.py: nearest shell entry ->
composite that one primitive evaluated at its peak -> advance -> repeat
until throughput dies). The difference is HOW the density is evaluated.

WHY THIS EXISTS. v5 and every other shader here read a precomputed
float32 inverse covariance matrix. That matrix cannot represent a
primitive whose axes differ by many orders of magnitude: for scales
(2.0, 2.0, 2e-6) the entries of Sigma^-1 span 1/4 to 2.5e11, and float32
carries about seven significant digits, so the in-plane curvature is lost
and the primitive degenerates into an effectively infinite slab. Measured
error in `power` against a float64 reference, by aspect ratio:

    aspect      matrix route      local-space route
        10         6.2e-08              2.1e-07
       100         7.6e-07              2.8e-07
      1000         6.0e-05              5.4e-07
     10000         6.4e-03              1.3e-06
    100000         6.2e-01              1.2e-06
   1000000         6.1e+01              1.6e-06

An error of 61 in `power` makes exp(power) meaningless. Trained 3DGS
scenes contain plenty of primitives in the 1e4 to 1e6 range, so this is
not a corner case, and it affects every estimator identically because
they all consume the same inv_covs tensor.

Condor et al. avoid this by construction. Their eval_transmission reads:

    o = rot.T * (ray.o - center) / scale
    d = rot.T * ray.d / scale
    t_peak = -dot(o, d) / dot(d, d)
    density = kernel.eval(ray(t_peak), ellipsoid)

The component-wise divide by `scale` handles each axis at its own
magnitude, so no ill-conditioned matrix is ever formed. This file does
the same thing. It is therefore both the numerically robust version AND
the one that matches their released implementation.

Algebraically A = dot(d_l, d_l), B = dot(o_l, d_l) and C = dot(o_l, o_l)
are the same quantities the matrix route computes, and the ray parameter
t is unchanged by the transform, since
    rot.T * ((x + t*w) - center) / scale = o_l + t * d_l.
So the 3-sigma shell entry is still the smaller root of
    A t^2 + 2B t + (C - K^2) = 0,
just evaluated on better-conditioned inputs.

NEW INPUT REQUIRED: `rotations`, a flat (N,4) float tensor of normalized
quaternions in (w, x, y, z) order, matching how rot_0..rot_3 are read
from the .ply. `inv_covs` and `covs` are no longer read by this shader.
Note that `covs` is still needed on the Python side, since
build_geometry_ads derives the AABB extents from it.

Everything else is v5 unchanged: alpha = min(opacity * exp(power),
0.9999), ordering by shell entry distance, primitives whose shell already
contains the ray origin are skipped (their BackfaceCulling semantics),
termination at throughput <= 0.01, lower-only clamp on emission.
*/

FORWARD {
    GPUPtr positions_ptr = load_tensor(parameters.positions);
    vec3_ptr positions = vec3_ptr(positions_ptr);
    GPUPtr colors_ptr = load_tensor(parameters.colors);
    vec3_ptr colors = vec3_ptr(colors_ptr);
    GPUPtr scales_ptr = load_tensor(parameters.scales);
    vec3_ptr scales = vec3_ptr(scales_ptr);
    GPUPtr rotations_ptr = load_tensor(parameters.rotations);
    float_ptr rotations = float_ptr(rotations_ptr);
    GPUPtr opacities_ptr = load_tensor(parameters.opacities);
    float_ptr opacities = float_ptr(opacities_ptr);
    GPUPtr f_rest_ptr = load_tensor(parameters.f_rest);
    float_ptr f_rest = float_ptr(f_rest_ptr);

    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = normalize(vec3(_input[3], _input[4], _input[5]));

    float sh_coefs[16];
    eval_sh(w, sh_coefs);

    const float K_SIGMA = 3.0;
    const int   MAX_DEPTH = 128;   // their render_3dg_asset.py setting; this is
                                    // the paper's Maximum Primitive Depth knob

    float T = 1.0;
    vec3 final_color = vec3(0.0);
    float t_cursor = 0.0;

    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        if (T <= 0.01) break;

        int   best_idx     = -1;
        float best_entry_t = 1e30;
        float best_alpha   = 0.0;

        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, accelerationStructureEXT(parameters.ads),
            gl_RayFlagsOpaqueEXT, 0xFF, x, t_cursor, w, 10000.0);

        while (rayQueryProceedEXT(rq)) {
            if (rayQueryGetIntersectionTypeEXT(rq, false) ==
                gl_RayQueryCandidateIntersectionAABBEXT) {

                int i = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);

                // --- quaternion -> rotation matrix rows, (w,x,y,z) order ---
                int qi = i * 4;
                float qr = rotations.data[qi + 0];
                float qx = rotations.data[qi + 1];
                float qy = rotations.data[qi + 2];
                float qz = rotations.data[qi + 3];

                vec3 R0 = vec3(1.0 - 2.0*(qy*qy + qz*qz),
                               2.0*(qx*qy - qr*qz),
                               2.0*(qx*qz + qr*qy));
                vec3 R1 = vec3(2.0*(qx*qy + qr*qz),
                               1.0 - 2.0*(qx*qx + qz*qz),
                               2.0*(qy*qz - qr*qx));
                vec3 R2 = vec3(2.0*(qx*qz - qr*qy),
                               2.0*(qy*qz + qr*qx),
                               1.0 - 2.0*(qx*qx + qy*qy));

                vec3 s = scales.data[i];
                vec3 d = x - positions.data[i];

                // --- R^T * v is the dot with each COLUMN of R ---
                vec3 o_l = vec3(dot(vec3(R0.x, R1.x, R2.x), d),
                                dot(vec3(R0.y, R1.y, R2.y), d),
                                dot(vec3(R0.z, R1.z, R2.z), d)) / s;
                vec3 d_l = vec3(dot(vec3(R0.x, R1.x, R2.x), w),
                                dot(vec3(R0.y, R1.y, R2.y), w),
                                dot(vec3(R0.z, R1.z, R2.z), w)) / s;

                float A = dot(d_l, d_l);
                if (A <= 1e-12) continue;
                float B = dot(o_l, d_l);
                float C = dot(o_l, o_l);

                // 3-sigma shell entry, same quadratic as before
                float disc = B*B - A*(C - K_SIGMA*K_SIGMA);
                if (disc < 0.0) continue;
                float t_enter = (-B - sqrt(disc)) / A;
                if (t_enter <= t_cursor) continue;   // origin already inside

                if (t_enter < best_entry_t) {
                    // peak, and the residual offset there, entirely in local space
                    float t_peak = -B / A;
                    vec3 off = o_l + t_peak * d_l;
                    float power = -0.5 * dot(off, off);   // <= 0 by construction
                    float alpha = min(opacities.data[i] * exp(power), 0.9999);
                    best_entry_t = t_enter;
                    best_idx = i;
                    best_alpha = alpha;
                }
            }
        }

        if (best_idx < 0) break;

        vec3 gaussian_color = colors.data[best_idx] * sh_coefs[0];
        int rest_idx = best_idx * 45;
        for (int c = 1; c < 16; ++c) {
            gaussian_color.x += f_rest.data[rest_idx + (c-1)     ] * sh_coefs[c];
            gaussian_color.y += f_rest.data[rest_idx + (c-1) + 15] * sh_coefs[c];
            gaussian_color.z += f_rest.data[rest_idx + (c-1) + 30] * sh_coefs[c];
        }
        gaussian_color = max(gaussian_color + 0.5, vec3(0.0));

        final_color += T * best_alpha * gaussian_color;
        T *= (1.0 - best_alpha);
        t_cursor = best_entry_t + 1e-4;
    }

    _output = float[](final_color.x, final_color.y, final_color.z);
}
BACKWARD {
    // Not implemented -- forward-only, consistent with this thesis's scope.
}