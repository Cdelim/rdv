/*
Estimator of the first hit depth 1 / (1 + d) and opacity (1 - transmittance) with RT
*/
#define SUBMAP_NAME extinction
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME majorant
#include "signatures/vec3_vec3_to_float_float.h"

#include "trait_transmittance_rt.h"

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
    _output[0] = 0.0;
    _output[1] = 0.0;

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    if (tMax <= 0 || tMin > tMax) // ray points away from the volume
        return;

    tMin = max(0, tMin);

    x += wo * tMin; // initial position in object space
    float d = tMax - tMin; // max_t wrt wo

    float T = 1.0;
    while (d > 0.) {
        float maj_distance;
        float maj = majorant(_this, x, w, maj_distance);
        float dt = min(maj_distance, -log(1.0 - random()) / maj); // sample free-flight distance
        x += dt * w;
        d -= dt;
        if (dt == maj_distance)
            continue; // no interaction withing slab, continue

        if (d <= 0.0)
            break; // reached the end of the segment

        float P = extinction(_this, x) / maj;

        if (_output[0] == 0.0) // no hit yet
            if (random() < P) // select this collision as the first hit
                _output[0] = 1.0 / (1.0 + tMax - d); // estimate of the first hit depth

        T *= (1.0 - P);
    }
    _output[1] = 1 - T; // opacity
}
