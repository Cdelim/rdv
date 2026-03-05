/*
parameters:
- density: Map vec3 -> float density
- majorant: Map vec3, vec3 -> float majorant, float distance
*/

#define SUBMAP_NAME density
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME majorant
#include "signatures/vec3_vec3_to_float_float.h"

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    float distance_scale = _input[7];
    float t = 0.0;
    while (t < d)
    {
        // get majorant in sub segment
        float majorant_val, segment_distance;
        majorant_val = distance_scale * majorant(_this, x + w * t, w, segment_distance);

        float dt = -log(1 - random()) / majorant_val;
        if (dt > segment_distance)
        {
            t += segment_distance;
            continue;
        }
        t += dt;
        float density_val = distance_scale * density(_this, x + w * t);
        if (random() < density_val / majorant_val)
        {
            _output[0] = t;
            return;
        }
    }
    _output[0] = d;
}