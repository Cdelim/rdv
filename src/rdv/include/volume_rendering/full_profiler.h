#define SUBMAP_NAME extinction
#include "signatures/vec3_to_float.h"

#include "trait_transmittance_rm.h"

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

    tMin = max(0, tMin);

    int current_step_index = 0;

    float tau = .0;

    float t = tMin + random() * parameters.step_size; // start from the middle of the first step for better accuracy
    float t_prev = tMin;
    float prev_extinction_val = 0.0;
    float last_t = 0;
    int count_samples = OUTPUT_DIM - 5; // OUTPUT_DIM = samples, transmittance(1), nee(1), light_dir(3)
    float first_t = POSINF;
    if (tMax > 0 && tMin <= tMax) // ray points away from the volume
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
            {
                first_t = (t_prev + dt);
                _output[current_step_index] = 1 / (0.1 + first_t);
            }
            else
                _output[current_step_index] = 1/(1 + 4 * pow((dt + t_prev - last_t), 0.5));
            last_t = t_prev + dt;
            current_step_index++;
        }
        t_prev = t;
        t += parameters.step_size;
        tau = next_tau;
        prev_extinction_val = extinction_val;
        if (tau > 50.0)
            break; // transmittance is too low, can stop early
    }
    _output[count_samples] = 1 - exp(-tau);
    // compute nee and light dir for the first sample
    _output[count_samples + 1] = 1.0;
    // concat relative light dir
    vec3_ptr rel_light_dir_ptr = vec3_ptr(parameters.relative_light_direction);
    _output[count_samples + 2] = rel_light_dir_ptr.data[0].x;
    _output[count_samples + 3] = rel_light_dir_ptr.data[0].y;
    _output[count_samples + 4] = rel_light_dir_ptr.data[0].z;
    if (first_t < POSINF)
    {
        vec3 sample_pos = x + wo * (first_t);
        vec3_ptr light_dir_ptr = vec3_ptr(parameters.light_direction);
        vec3 light_dir_object_space = L * light_dir_ptr.data[0]; // convert light direction to object space
        ray_box_intersection(sample_pos, light_dir_object_space, tMin, tMax);
        if (tMax <= 0 || tMin > tMax) // should not happend but just in case
            return;
        _output[count_samples + 1] = transmittance_rm(_this, sample_pos, light_dir_object_space, tMax, parameters.step_size);
    }
}