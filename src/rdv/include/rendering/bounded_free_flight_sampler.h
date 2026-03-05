/*
parameters:
- base_transmittance: ray segment + sigma_scale -> float transmittance
- boundary: ray segment -> tnext
- bmin, bmax: vec3 - bounding box corners
*/

#define SUBMAP_NAME boundary
#include "signatures/vec3_vec3_to_float.h"

#define SUBMAP_NAME base_free_flight_sampler
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
    _output[0] = _input[6];
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
            float st = base_free_flight_sampler(_this, current_x * scale + offset, wt, distance * distance_scale, _input[7] / distance_scale) / distance_scale;
            if (st < distance) {
                _output[0] = t + st;
                return;
            }
        }
        t += distance + 0.00001;
    }
}

BACKWARD {
}