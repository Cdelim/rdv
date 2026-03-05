#define SUBMAP_NAME extinction
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME majorant
#include "signatures/vec3_vec3_to_float_float.h"

#include "trait_transmittance_rt.h"

#define SUBMAP_NAME anisotropy
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME scattering_albedo
#include "signatures/vec3_to_spectral.h"

#define SUBMAP_NAME environment
#include "signatures/vec3_to_spectral.h"

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    // Transform ray to local space
    mat4x3 M = mat4x3_ptr(load_tensor(parameters.transform)).data[0]; // from object to world space
    mat3 L = inverse(mat3(M[0].xyz, M[1].xyz, M[2].xyz));
    vec3 O = M[3].xyz;
    x = L * (x - O); // convert world position to object space position (use same x)
    vec3 wo = L * w; // convert world direction to object space direction
    // notice, in this point wo is not normalized, but is not a problem since we will use it for traversing the unnormalized density field, and we will normalize it later for sampling the phase function
    for (int i=0; i<SPECTRAL_DIM; i++)
        _output[i] = 0.0;

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    if (tMax <= 0 || tMin > tMax) // ray points away from the volume
    {
#ifdef ENVIRONMENT
        environment(_this, w, _output); // direct environment lighting
#endif
        return;
    }

    x += wo * tMin; // initial position in object space
    float d = tMax - tMin; // max_t wrt wo

    float W[SPECTRAL_DIM]; // path throughput, accumulation is direct to output to save register, so we need to store it in a local variable
    for (int i=0; i<SPECTRAL_DIM; i++)
        W[i] = 1.0;
    int bounces = 0;
    while (d > 0) {
        float maj_distance;
        float maj = majorant(_this, x, wo, maj_distance);
        float dt = min(maj_distance, -log(1.0 - random()) / maj); // sample free-flight distance
        x += dt * wo;
        d -= dt;
        if (dt == maj_distance)
            continue; // no interaction withing slab, continue
        if (d <= 0.0)
            break; // reached the end of the segment
        if (random() >= extinction(_this, x) / maj) // null collision
            continue;

        // -- Valid collision --
        // modulate path-throughput with scattering albedo
        float sa[SPECTRAL_DIM];
        scattering_albedo(_this, x, sa); // compute scattering albedo
        for (int i=0; i<SPECTRAL_DIM; i++)
            W[i] *= sa[i];

        bool some_throughput = false;
        for (int i=0; i<SPECTRAL_DIM; i++)
            if (W[i] > 0.0001)
                some_throughput = true;
        if (!some_throughput)
            return; // early termination if path throughput is too low

        // get anisotropy for phase function sampling
        float g = anisotropy(_this, x);

#ifdef ENVIRONMENT_SAMPLER
        if (bounces > 1)
        {
            // add contribution from environment lighting with NEE
            float env_sample[3 + SPECTRAL_DIM + 1]; // wnee, env/pdf(wnee), pdf (pdf not necessary here)
            // x is in object space, w is in world space, we need to convert x to world space for querying environment sampler, and convert nee_w back to object space for computing transmittance
            forward(parameters.environment_sampler, float[](x.x, x.y, x.z, w.x, w.y, w.z), env_sample);
            vec3 nee_w = vec3(env_sample[0], env_sample[1], env_sample[2]);
            float nee_phase = hg_phase_eval(w, nee_w, g);
            nee_w = L * nee_w; // convert to object space
            float nee_d;
            ray_box_intersection(x, nee_w, tMin, nee_d); // compute intersection with volume boundary for next iteration
            float tr = transmittance_rt(_this, x, nee_w, nee_d);
            tr *= nee_phase;
            for (int i=0; i<SPECTRAL_DIM; i++)
                _output[i] += W[i] * env_sample[3 + i] * tr; // accumulate NEE contribution
        }
#endif
        // scatter ray
        w = hg_phase_sample(w, g); // compute scattering direction in world space
        wo = L * w; // convert scattering direction to object space for next iteration of ray marching
        // compute next distance
        ray_box_intersection(x, wo, tMin, d); // compute intersection with volume boundary for next iteration
        bounces ++;
    }

#ifdef ENVIRONMENT
    float env[SPECTRAL_DIM];
    environment(_this, w, env); // add environment contribution if ray escaped the volume
    for (int i=0; i<SPECTRAL_DIM; i++)
        _output[i] += W[i] * env[i];
#endif
}
