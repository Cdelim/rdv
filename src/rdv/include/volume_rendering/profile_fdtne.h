#define SUBMAP_NAME extinction
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME majorant
#include "signatures/vec3_vec3_to_float_float.h"

#define SUBMAP_NAME anisotropy
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME scattering_albedo
#include "signatures/vec3_to_spectral.h"

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

    for (int i=0; i < OUTPUT_DIM; i++)
        _output[i] = 0.0; // transmittance 1 till infinity

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);

    tMin = max(0, tMin);

    int current_step_index = 0;

    float tau = .0;

    float t = tMin + parameters.step_size * random(); // start from the middle of the first step for better accuracy
    float t_prev = tMin;
    float prev_extinction_val = 0.0;
    float last_t = 0;
    int count_samples = DEPTH_SAMPLES; // OUTPUT_DIM = samples, transmittance(1)
    float check_t = POSINF;
    if (tMax > 0 && tMin <= tMax) // ray points away from the volume

    while (t_prev < tMax)
    {
        float extinction_val = extinction(_this, x + wo * (t_prev + t)*0.5); // sample extinction at the middle of the step for better accuracy
        float next_tau = tau + extinction_val * parameters.step_size;
        while (current_step_index < count_samples && parameters.transmittance_threshold[current_step_index] < next_tau)
        {
            float T = parameters.transmittance_threshold[current_step_index] - tau;
            float dt = T / extinction_val;
            if (current_step_index == 0) // first segment treated differently
            {
                check_t = (t_prev + dt);
                _output[current_step_index] = 1 / (0.1 + check_t);
            }
            else
                _output[current_step_index] = 1/(1 + 4 * pow((t_prev + dt - last_t), 0.5));
            last_t = t_prev + dt;
            current_step_index++;
        }
        t_prev = t;
        t += parameters.step_size;
        tau = next_tau;
        if (tau > 100.0)
            break; // transmittance is too low, can stop early
    }
    // hyperbolic function to map tau to (0, 1), can be replaced by other functions, e.g., 1 - exp(-tau)
    _output[count_samples] = 1 - 1 / (1 + tau*0.5); // 1 - exp(-tau);
    // compute nee and light dir for the first sample
    if (check_t != POSINF)
    {
        vec3 sample_pos = x + wo * (check_t); // position of the last sample in object space
        // sample light source for the first sample
        float g = anisotropy(_this, sample_pos);
        float sa[SPECTRAL_DIM];
        scattering_albedo(_this, sample_pos, sa);
        float env_sampler[3 + SPECTRAL_DIM + 1];
        forward(parameters.environment_sampler, float[6](sample_pos.x, sample_pos.y, sample_pos.z, w.x, w.y, w.z), env_sampler);
        vec3 light_dir = vec3(env_sampler[0], env_sampler[1], env_sampler[2]); // in world space
        float rho_nee = hg_phase_eval(w, light_dir, g);
        vec3 light_dir_object_space = L * light_dir; // convert light direction to object space
        ray_box_intersection(sample_pos, light_dir_object_space, tMin, tMax);
        float nee_T = transmittance_rt(_this, sample_pos, light_dir_object_space, tMax);
        for (int i=0; i<SPECTRAL_DIM; i++)
            // nee contribution
            _output[count_samples + 1 + i] = env_sampler[3 + i] * rho_nee * sa[i] * nee_T; // env * phase * albedo * transmittance
        // sample environment wrt phase
        light_dir = hg_phase_sample(w, g); // importance sample the environment map according to the phase function
        float env[SPECTRAL_DIM];
        forward(parameters.environment, float[3](light_dir.x, light_dir.y, light_dir.z), env);
        light_dir_object_space = L * light_dir; // convert light direction to object space
        ray_box_intersection(sample_pos, light_dir_object_space, tMin, tMax);
        nee_T = transmittance_rt(_this, sample_pos, light_dir_object_space, tMax);
        for (int i=0; i<SPECTRAL_DIM; i++)
            // nee contribution
            _output[count_samples + 1 + SPECTRAL_DIM + i] = env[i] * sa[i] * nee_T; // env * phase * albedo * transmittance
        for (int i=0; i<SPECTRAL_DIM; i++)
            _output[count_samples + 1 + 2 * SPECTRAL_DIM + i] = sa[i]; // albedo
        _output[count_samples + 1 + 3 * SPECTRAL_DIM] = g; // anisotropy for better visualization
    }
}