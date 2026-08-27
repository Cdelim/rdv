#define SUBMAP_NAME extinction
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

    for (int i=0; i < OUTPUT_DIM; i++)
        _output[i] = 0.0; // transmittance 1 till infinity

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    if (tMax <= 0 || tMin > tMax) // ray points away from the volume
        return;

    tMin = max(0, tMin);

    int current_step_index = 0;

    float T = 1.0;

    float t = tMin + 0.5 * parameters.step_size; // start from the middle of the first step for better accuracy
    float t_prev = 0;
    while (t < tMax)
    {
        float extinction_val = extinction(_this, x + wo * t); // sample extinction at the middle of the step for better accuracy
        T *= exp(-extinction_val * parameters.step_size);
        while (current_step_index < OUTPUT_DIM - 1 && T < parameters.transmittance_threshold[current_step_index])
        {
            _output[current_step_index] = 1/(1 + sqrt(t - t_prev));
            t_prev = t;
            current_step_index++;
        }
        #ifndef PRODUCE_TRANSMITTANCE
        if (current_step_index >= OUTPUT_DIM - 1)
            break; // all profiling points are filled, can stop early
        #endif
        if (T < 0.001)
            break; // transmittance is too low, can stop early
        t += parameters.step_size;
    }
    #ifdef PRODUCE_TRANSMITTANCE
    _output[OUTPUT_DIM - 1] = T;
    #endif
}