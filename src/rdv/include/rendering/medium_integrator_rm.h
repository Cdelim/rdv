#define SUBMAP_NAME density
#include "signatures/vec3_to_float.h"

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    vec3 xf = x + w * d;
    // Set final position and direction in output
    _output[2*SPECTRAL_DIM + 0] = xf.x; // xout
    _output[2*SPECTRAL_DIM + 1] = xf.y;
    _output[2*SPECTRAL_DIM + 2] = xf.z;
    _output[2*SPECTRAL_DIM + 3] = w.x;
    _output[2*SPECTRAL_DIM + 4] = w.y;
    _output[2*SPECTRAL_DIM + 5] = w.z;

    Tensor t = load_deferred(parameters.transform);
    mat4x3 T = mat4x3_ptr(t.data_ptr).data[0];
    transform_ray_to_object(x, w, T);
    float density_scale = 1 / length(w);
    w *= density_scale; // normalize w
    d /= density_scale;
    int samples = int(ceil(d * density_scale / parameters.step_size)); // samples are taken according to distance in world space
    float tau = 0.0; // tau
    float dt = d / float(samples);
    for (int i = 0; i < samples; i++) {
        tau += density(_this, x + w * dt * (i + 0.5));
    }
    float transmittance = exp(-tau * (d * density_scale / float(samples)));
    /* output is A, W, xout, wout */
    for (int i=0; i<SPECTRAL_DIM; i++)
    {
        _output[i] = 0.0; // accumulated radiance (no emission from now)
        _output[SPECTRAL_DIM + i] = transmittance; // throughput
    }
}

BACKWARD {
}