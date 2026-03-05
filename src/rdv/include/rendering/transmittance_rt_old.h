#define MAP_NAME density
#include "signatures/vec3_to_float.h"

#define MAP_NAME boundary
#include "signatures/vec3_vec3_to_float.h"

#define MAP_NAME majorant
#include "signatures/vec3_vec3_to_float_float.h"


FORWARD {
    // start ray
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    _output[0] = 1.0;
    while(true)
    {
        float distance = boundary(_this, x, w);
        if (distance == POSINF || distance == NEGINF)
            return;
        if (distance >= 0.0) // outside geometry
        {
            x += w*(distance + 0.0001); // move a bit further
            continue;
        }
        // inside geometry
        float t_exit = -distance;
        float t = 0.0;
        while (t < t_exit)
        {
            float density_val = density(_this, x + w * t);
            _output[0] *= exp(-density_val * parameters.step_size);
            t += parameters.step_size;
        }
        x += w * (t_exit + 0.0001);
    }
}

