/*
Computes the transmittance along a ray segment inside a participating medium.
parameters:
medium_transmittance: Map segment + d -> float transmittance
boundary: Map ray -> surfel
*/

#define SUBMAP_NAME boundary
#include "signatures/vec3_vec3_to_float.h"

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    float distance_scale = _input[7];

    float hit_d = boundary(_this, x, w);

    if (hit_d < d) // the ray hits the boundary before reaching distance d or is inside
    {
        _output[0] = 0.0; // fully occluded
        return;
    }

    forward(parameters.medium_transmittance, _input, _output);
}

BACKWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    float distance_scale = _input[7];

    float hit_d = boundary(_this, x, w);

    if (hit_d < d) // the ray hits the boundary before reaching distance d
        return; // no gradient since transmittance is 0
#ifndef BW_USES_OUTPUT
#ifdef INPUT_REQUIRES_GRAD
    backward(parameters.medium_transmittance, _input, _output_grad, _input_grad);
#else
    backward(parameters.medium_transmittance, _input, _output_grad);
#endif
#else
#ifdef INPUT_REQUIRES_GRAD
    backward(parameters.medium_transmittance, _input, _output, _output_grad, _input_grad);
#else
    backward(parameters.medium_transmittance, _input, _output, _output_grad);
#endif
#endif
}