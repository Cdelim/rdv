#define SUBMAP_NAME integrand
#include "signatures/vec3_to_spectral.h"


FORWARD {
    vec3 x0 = vec3(_input[0], _input[1], _input[2]);
    vec3 x1 = vec3(_input[3], _input[4], _input[5]);
    vec3 x = x0;
    vec3 w = x1 - x0; // ray direction from x0 to x1, not normalized, but is not a problem since we will use it for traversing the unnormalized density field, and we will normalize it later for sampling the phase function
    float d = length(w);
    if (d < 0.0000001) // ray is too short, treat it
    {
        _output[0] = 0.0;
        return;
    }
    w /= d; // normalize w to get the correct ray direction, and we will use d for traversing the density field

    // Transform ray to local space
    mat4x3 M = mat4x3_ptr(load_tensor(parameters.transform)).data[0]; // from object to world space
    mat3 L = inverse(mat3(M[0].xyz, M[1].xyz, M[2].xyz));
    vec3 O = M[3].xyz;
    x = L * (x - O); // convert world position to object space position (use same x)
    vec3 wo = L * w; // convert world direction to object space direction
    // notice, in this point wo is not normalized, but is not a problem since we will use it for traversing the unnormalized density field, and we will normalize it later for sampling the phase function

    for (int i=0; i < OUTPUT_DIM; i++)
        _output[i] = 0.0; // zero integral by default

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    if (tMax <= 0 || tMin > tMax) // ray points away from the volume
        return;

    tMin = max(0, tMin);

    x += wo * tMin; // initial position in object space
    d = tMax - tMin; // max_t wrt wo

    x += parameters.step_size * w * random(); // start from the middle of the first step for better accuracy
    while (d > 0.) {
        float current_sample[OUTPUT_DIM];
        integrand(_this, x, current_sample);
        for (int i=0; i<OUTPUT_DIM; i++)
            _output[i] += current_sample[i] * parameters.step_size; // accumulate integral using the rectangle rule
        d -= parameters.step_size;
        x += parameters.step_size * wo;
    }
}

