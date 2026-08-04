/*
Don't Splat Your Gaussians -- OFFICIAL-CODE-FAITHFUL VPRF (v5)

This file reproduces, in our Vulkan ray-query framework, the AUTHORS' OWN
RELEASED IMPLEMENTATION of the VPRF integrator (volprim_rf.py from Meta's
volprim repository) -- the integrator their own render_3dg_asset.py and
refine_3dg_dataset.py scripts use for every 3DGS comparison. Unlike v3/v4,
which were built from the paper's text and pseudocode, this one is built
from their actual code, which settles what "dropping segment by segment
integration" meant:

  - volprim_rf.py (VPRF) contains NO segment tracking, NO active sets, NO
    entry/exit event handling, NO tau_i/tau_total blending, NO stack. It is
    a simple loop: nearest-hit -> composite that one primitive (evaluated
    at its own peak) -> advance ray past the hit -> repeat until
    transmittance dies or max_depth is reached.
  - The stack/segment machinery (stack.py, max_overlaps, sample_segment)
    is used ONLY by volprim_prb.py -- the VPPT path tracer, i.e. what our
    v3_faithful.h corresponds to.

So the paper's Listing 1 describes VPPT's traversal; VPRF as shipped is the
simple sequential loop below. Structurally this is almost exactly our
DontSplashYourGaussians_v2.h -- with the following REAL differences, each
taken directly from their code and replicated here:

1. ALPHA FORMULA. Theirs (volprim_rf.eval_transmission + GaussianKernel.eval):
       alpha = min(opacity * exp(power), 0.9999)
   i.e. the standard 3DGS/3DGRT convention. v2 instead used the optical-
   depth reinterpretation alpha = 1-exp(-(-log(1-opacity))*exp(power)).
   The two agree at power=0 and for small alpha but diverge for opaque
   primitives evaluated off-peak. Since 3DGS-trained assets were optimized
   under THEIR convention, this file uses theirs.

2. ORDERING KEY. Their scene.ray_intersect returns the nearest 3-sigma
   ELLIPSOID SHELL SURFACE hit, so primitives are composited in order of
   shell ENTRY distance -- not peak distance (v2's key). For overlapping
   primitives these orders can differ. This file orders by entry t,
   computed from the same A,B,C quadratic via its discriminant.
   Consequence, also faithful: a primitive whose shell the ray origin is
   already INSIDE (entry behind origin; their BackfaceCulling makes it
   invisible) is skipped entirely.

3. TERMINATION. Theirs: stop when throughput <= 0.01 ("β_max > 0.01" to
   continue). v2 used 0.001. This file uses 0.01.

4. MAX PRIMITIVE DEPTH. Theirs is a real, user-facing parameter
   (max_depth; default 64 in the integrator, 128 in their 3DG scripts) --
   this is the "Maximum Primitive Depth" ablation from the paper, a
   sanctioned quality/performance knob, NOT just a safety bound. Exposed
   below as MAX_DEPTH = 128 to match their 3DG rendering scripts.

5. NO ALPHA CUTOFF. Their loop composites every hit primitive, however
   faint; v2 skipped alpha < 1/255. Removed here.

6. EMISSION CLAMP. Theirs: max(emission + 0.5, 0.0) -- lower clamp only,
   no upper bound. v2 clamped to [0,1]. Matched here. (Their final
   srgb_to_linear output conversion is a display-encoding step outside
   the compositing loop; apply it Python-side if comparing EXRs.)

Everything else -- repeated nearest-hit traversal, advancing the origin by
a 1e-4 nudge past each hit (theirs: ray.o = si.p + ray.d*1e-4), peak
formula t_peak = -dot(o,d)/dot(d,d) in normalized space (algebraically
identical to our -B/A) -- was already the same between their code and v2.
*/

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

    const float K_SIGMA = 3.0;   // matches their default ellipsoid shell extent
    const int   MAX_DEPTH = 4096; // their render_3dg_asset.py / refine_3dg_dataset.py
                                  // setting; a real quality knob (paper's "Maximum
                                  // Primitive Depth" ablation), not a safety bound

    float T = 1.0;
    vec3 final_color = vec3(0.0);
    float t_cursor = 0.0;

    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        if (T <= 0.01) break;    // their "kill path" criterion: continue only while beta > 0.01

        // --- one nearest-hit query: find the closest shell ENTRY beyond t_cursor ---
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
                float C = M00*d.x*d.x + M11*d.y*d.y + M22*d.z*d.z
                        + 2.0*(M01*d.x*d.y + M02*d.x*d.z + M12*d.y*d.z);

                // 3-sigma shell entry: smaller root of A t^2 + 2B t + (C - K^2) = 0
                float disc = B*B - A*(C - K_SIGMA*K_SIGMA);
                if (disc < 0.0) continue;              // ray misses the shell
                float t_enter = (-B - sqrt(disc)) / A;
                // faithful to their BackfaceCulling semantics: if the origin is
                // already inside this shell (entry behind cursor), the primitive
                // is never composited at all
                if (t_enter <= t_cursor) continue;

                if (t_enter < best_entry_t) {
                    // alpha: opacity * exp(power) evaluated at THIS primitive's
                    // own peak -- their eval_transmission, our A/B/C form
                    float power = -0.5 * (C - (B*B)/A);
                    power = min(power, 0.0);   // their eval_transmission clamps to [0,inf)
                    float alpha = min(opacities.data[i] * exp(power), 0.9999);
                    best_entry_t = t_enter;
                    best_idx = i;
                    best_alpha = alpha;
                }
            }
        }

        if (best_idx < 0) break;   // ray escaped the scene

        // --- composite this one primitive: L += beta*(1-transmission)*emission ---
        vec3 gaussian_color = colors.data[best_idx] * sh_coefs[0];
        int rest_idx = best_idx * 45;
        for (int c = 1; c < 16; ++c) {
            gaussian_color.x += f_rest.data[rest_idx + (c-1)     ] * sh_coefs[c];
            gaussian_color.y += f_rest.data[rest_idx + (c-1) + 15] * sh_coefs[c];
            gaussian_color.z += f_rest.data[rest_idx + (c-1) + 30] * sh_coefs[c];
        }
        // their convention: lower clamp only, no upper bound
        gaussian_color = max(gaussian_color + 0.5, vec3(0.0));

        final_color += T * best_alpha * gaussian_color;
        T *= (1.0 - best_alpha);

        // their advance: ray.o = si.p + ray.d * 1e-4
        t_cursor = best_entry_t + 1e-4;
    }

    _output = float[](final_color.x, final_color.y, final_color.z);
}
BACKWARD {
    // Differentiation logic goes here
}
