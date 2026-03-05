#define SUBMAP_NAME extinction
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME majorant
#include "signatures/vec3_vec3_to_float_float.h"

#define SUBMAP_NAME anisotropy
#include "signatures/vec3_to_float.h"

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

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);

    x += wo * tMin; // initial position in object space
    float d = tMax <= 0.0 ? 0.0 : tMax - tMin; // max_t wrt wo

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

        // get anisotropy for phase function sampling
        float g = anisotropy(_this, x);
        // scatter ray
        w = hg_phase_sample(w, g); // compute scattering direction in world space
        wo = L * w; // convert scattering direction to object space for next iteration of ray marching
        // compute next distance
        ray_box_intersection(x, wo, tMin, d); // compute intersection with volume boundary for next iteration
        bounces ++;
    }

    _output[0] = w.x;
    _output[1] = w.y;
    _output[2] = w.z;
    _output[3] = float(bounces); // scattering
}
