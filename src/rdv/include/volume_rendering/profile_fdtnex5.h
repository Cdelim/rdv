#define SUBMAP_NAME extinction
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME majorant
#include "signatures/vec3_vec3_to_float_float.h"

#define SUBMAP_NAME anisotropy
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME scattering_albedo
#include "signatures/vec3_to_spectral.h"

#include "trait_transmittance_rt.h"
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
    if (tMax > 0 && tMin <= tMax) // ray points away from the volume
    {
        // compute tau
        float tau = .0;
        float t = tMin;
        while (t < tMax)
        {
            float extinction_val = extinction(_this, x + wo * t); // sample extinction at the middle of the step for better accuracy
            tau += extinction_val * parameters.step_size;
            t += parameters.step_size;
//            if (tau > 100.0)
//                break; // transmittance is too low, can stop early
        }

        t = tMin + parameters.step_size * random();  // restart raymarching
        float current_tau = 0.0;
        int current_depth_index  = 0;
        float scale = parameters.transmittance_threshold[current_depth_index];
        float tau_crossing = scale * tau < 0.01 ? 0.5 * scale * tau : scale * tau > 20 ? 1 : 1 - (scale * tau * exp(-scale * tau))/(1 - exp(-scale * tau)); // expected tau when crossing the transmittance threshold, derived from the exponential distribution of free-flight distance
        if (tau > 0.0001)
            while (t < tMax)
            {
                float extinction_val = extinction(_this, x + wo * t); // sample extinction at the middle of the step for better accuracy
                current_tau += extinction_val * parameters.step_size;
                while (current_tau * scale >= tau_crossing)
                {
                    _output[current_depth_index] = t; // 1 - 1 / (1 + current_tau*0.5); // 1 - exp(-current_tau);
                    current_depth_index++;
                    if (current_depth_index >= DEPTH_SAMPLES)
                        break;
                    scale = parameters.transmittance_threshold[current_depth_index];
                    tau_crossing = scale * tau < 0.01 ? 0.5 * scale * tau : scale * tau > 20 ? 1 : 1 - (scale * tau * exp(-scale * tau))/(1 - exp(-scale * tau));
                }
                if (current_depth_index >= DEPTH_SAMPLES)
                        break;
                t += parameters.step_size;
    //            if (current_tau > 100.0)
    //                break; // transmittance is too low, can stop early
            }

        // compute nee and light dir for the first sample
        if (current_depth_index > 0) // if we have at least one valid sample, we can compute nee and scattering for the first sample
        {
            float check_t = _output[1]; // index to the depth sample that we want to check for nee, scattering, etc.

            // reencode depths
            for (int i=DEPTH_SAMPLES-1; i >= 1; i--)
                _output[i] = 1/(1 + 4 * pow(_output[i] - _output[i-1], 0.5));
            _output[0] = 1 / (0.1 + _output[0]);

            vec3 sample_pos = x + wo * check_t; // position of the last sample in object space
            // sample light source for the first sample
            float g = anisotropy(_this, sample_pos);
            // feature: transmittance on a reduced medium
            _output[DEPTH_SAMPLES] = 1 - exp(-0.2 * tau); // 1 - 1 / (1 + tau*0.5); // 1 - exp(-tau);
            float sa[SPECTRAL_DIM];
            scattering_albedo(_this, sample_pos, sa);
            float env_sampler[3 + SPECTRAL_DIM + 1];
            forward(parameters.environment_sampler, float[6](sample_pos.x, sample_pos.y, sample_pos.z, w.x, w.y, w.z), env_sampler);
            vec3 light_dir = vec3(env_sampler[0], env_sampler[1], env_sampler[2]); // in world space
            float rho_nee = hg_phase_eval(w, light_dir, g);
            vec3 light_dir_object_space = L * light_dir; // convert light direction to object space
            ray_box_intersection(sample_pos, light_dir_object_space, tMin, tMax);
            float nee_T = tMax < 0 || tMax < tMin ? 1.0 : transmittance_rm(_this, sample_pos, light_dir_object_space, tMax, parameters.step_size, 1.0);
//            float nee_T = tMax < 0 || tMax < tMin ? 1.0 : transmittance_rt(_this, sample_pos, light_dir_object_space, tMax, 1.0);
            for (int i=0; i<SPECTRAL_DIM; i++)
                // nee contribution
                _output[DEPTH_SAMPLES + 1 + i] = (1 - exp(-tau * (1 - g))) * env_sampler[3 + i] * nee_T * rho_nee; // env * phase * albedo * transmittance
            // sample environment wrt phase
            int env_samples = parameters.environment_samples;
            for (int s=0; s <  env_samples; s++)
            {
                // uniform sampling
                light_dir = hg_phase_sample(w, g); // importance sample the environment map according to the phase function
                float env[SPECTRAL_DIM];
                forward(parameters.environment, float[3](light_dir.x, light_dir.y, light_dir.z), env);
                light_dir_object_space = L * light_dir; // convert light direction to object space
                ray_box_intersection(sample_pos, light_dir_object_space, tMin, tMax);
                nee_T = transmittance_rt(_this, sample_pos, light_dir_object_space, tMax, (1 - g));
                for (int i=0; i<SPECTRAL_DIM; i++)
                    // nee contribution
                    _output[DEPTH_SAMPLES + 1 + SPECTRAL_DIM + i] += (1 - exp(-tau * (1 - g))) * nee_T * env[i] / env_samples; // env * phase * albedo * transmittance
            }
            for (int i=0; i<SPECTRAL_DIM; i++)
                _output[DEPTH_SAMPLES + 1 + 2 * SPECTRAL_DIM + i] = pow(sa[i], 2*tau*(1-g));// 1 / (1 - log(1-sa[i])); // albedo
            _output[DEPTH_SAMPLES + 1 + 3 * SPECTRAL_DIM] = g; // anisotropy
        }
    }
}