#define SUBMAP_NAME density
#include "signatures/vec3_to_float.h"

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    Tensor t = load_deferred(parameters.transform);
    mat4x3 T = mat4x3_ptr(t.data_ptr).data[0];
    transform_ray_to_object(x, w, T);
    float density_scale = 1 / len(w);
    w *= density_scale; // normalize w
    d /= density_scale;
    int samples = int(ceil(d * density_scale / parameters.step_size)); // samples are taken according to distance in world space
    _output[0] = 0.0; // tau
    float dt = d / float(samples);
    for (int i = 0; i < samples; i++) {
        float sigma = density(_this, x + w * dt * (i + 0.5));
        _output[0] -= sigma;
    }
    _output[0] = exp(_output[0] * (d * density_scale / float(samples)));
}

BACKWARD {
#ifdef INPUT_REQUIRES_GRAD
    NOT_SUPPORTED("[ERROR] Not supported transmittance prop to input.");
    return;
#endif
#ifndef BW_USES_OUTPUT
    float _output[1];
    #ifdef STOCHASTIC
    uvec4 seed = random_seed();
    #endif
    forward(_this, _input, _output);
    #ifdef STOCHASTIC
    random_seed(seed); // restore seed before forward
    #endif
#endif
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    float density_scale = _input[7];
    int samples = int(ceil(d * density_scale / parameters.step_size)); // samples are taken according to distance in world space
    float dt = d / float(samples);
    float T_dL_dT = _output[0] * _output_grad[0];
    float dL_dsigma = - T_dL_dT * d * density_scale / samples;
    for (int i = 0; i < samples; i++)
        density_bw(_this, x + w * dt * (i + 0.5), dL_dsigma);
}