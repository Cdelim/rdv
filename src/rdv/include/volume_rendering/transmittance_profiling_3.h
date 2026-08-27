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

    float tau = .0;

    float t = tMin + parameters.step_size; // start from the middle of the first step for better accuracy
    float t_prev = tMin;
    float prev_extinction_val = 0.0;
    float last_t = 0;
    #ifdef PRODUCE_TRANSMITTANCE
    int count_samples = OUTPUT_DIM - 1;
    #else
    int count_samples = OUTPUT_DIM;
    #endif
    while (t < tMax)
    {
        float extinction_val = extinction(_this, x + wo * t); // sample extinction at the middle of the step for better accuracy
        float next_tau = tau + (extinction_val + prev_extinction_val) * parameters.step_size * 0.5; // trapezoidal rule for better accuracy
        float m = (extinction_val - prev_extinction_val) / parameters.step_size; // slope of extinction between t_prev and t
        float a = prev_extinction_val;
        while (current_step_index < count_samples && parameters.transmittance_threshold[current_step_index] < next_tau)
        {
            float T = parameters.transmittance_threshold[current_step_index] - tau;
            float D = a*a + 2*m*T;
            float dt;
            if (m == 0)
                dt = T / a;
            else
                dt = (-a + sqrt(D)) / m;//max((-a - sqrt(D)) / m, (-a + sqrt(D)) / m);
            if (current_step_index == 0) // first segment treated differently
                _output[current_step_index] = 1 / (0.1 + t_prev + dt);
            else
                _output[current_step_index] = 1/(1 + 4 * pow(dt + t_prev - last_t, 0.5));
            last_t = t_prev + dt;
            current_step_index++;
        }
        t_prev = t;
        t += parameters.step_size;
        tau = next_tau;
        prev_extinction_val = extinction_val;
        #ifndef PRODUCE_TRANSMITTANCE
        if (current_step_index >= OUTPUT_DIM - 1)
            break; // all profiling points are filled, can stop early
        #endif
        if (tau > 50.0)
            break; // transmittance is too low, can stop early
    }
    #ifdef PRODUCE_TRANSMITTANCE
    _output[OUTPUT_DIM - 1] = 1 - exp(-tau);
    #endif
}