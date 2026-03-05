/*
parameters:
- base_transmittance: ray segment + sigma_scale -> float transmittance
- boundary: ray segment -> tnext
  If the ray does not intersect the bounding box, transmittance is 1.0
- bmin, bmax: vec3 - bounding box corners
*/

#define SUBMAP_NAME boundary
#include "signatures/vec3_vec3_to_float.h"

#define SUBMAP_NAME base_transmittance
#define SUBMAP_BW_USES_OUTPUT
#include "signatures/vec3_vec3_float_float_to_float.h"

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 scale = 2.0 / (parameters.bmax - parameters.bmin);
    vec3 offset = -0.5 * scale * (parameters.bmin + parameters.bmax);
    vec3 wt = scale * w;
    float distance_scale = length(wt);
    wt /= distance_scale;
    _output[0] = 1.0;
    float t = 0;
    while (t < _input[6]){
        vec3 current_x = x + w * t;
        float distance = boundary(_this, current_x, w);
        if (distance == POSINF) // no more intersection
            return;
        if (distance < 0.0) // outside geometry
        {
            // inside geometry
            distance = min(_input[6] - t, -distance);
            _output[0] *= base_transmittance(_this, current_x * scale + offset, wt, distance * distance_scale, _input[7] / distance_scale);
        }
        t += distance + 0.00001;
    }
}

BACKWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 scale = 2.0 / (parameters.bmax - parameters.bmin);
    vec3 offset = -0.5 * scale * (parameters.bmin + parameters.bmax);
    vec3 wt = scale * w;
    float distance_scale = length(wt);
    wt /= distance_scale;

#ifndef BW_USES_OUTPUT
#ifdef STOCHASTIC
    uvec4 seed = random_seed();
#endif
    float _output[1];
    forward(_this, _input, _output);
#endif
#ifdef STOCHASTIC
    random_seed(seed); // restore seed before forward
#endif
    float T_dL_dT = _output[0] * _output_grad[0];
    float t = 0;
    while (t < _input[6]){
        vec3 current_x = x + w * t;
        float distance = boundary(_this, current_x, w);
        if (distance == POSINF) // no more intersection
            return;
        if (distance < 0.0) // outside geometry
        {
            // inside geometry
            distance = min(_input[6] - t, -distance);
            base_transmittance_bw(_this, current_x * scale + offset, wt, distance * distance_scale, _input[7] / distance_scale, 1.0, T_dL_dT); // output 1 since T already multiplied in gradient.
        }
        t += distance + 0.00001;
    }
}