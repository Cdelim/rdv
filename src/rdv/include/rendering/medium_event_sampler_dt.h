#define SUBMAP_NAME extinction
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME majorant
#include "signatures/vec3_vec3_to_float_float.h"

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6]; // max_t

    // Transform ray to local space
    Tensor T = load_deferred(parameters.transform);
    mat4x3 M = mat4x3_ptr(T.data_ptr).data[0];
    transform_ray_to_object(x, w, M);
    float density_scale = 1 / length(w);
    w *= density_scale; // normalize w
    d /= density_scale;

    float t = 0;
    while (t < d) {
        float maj_distance;
        float maj = majorant(_this, x + w * t, w, maj_distance);
        float dt = -log(1.0 - random()) / (maj * density_scale); // sample free-flight distance
        if (dt > maj_distance){
            t += maj_distance;
            continue; // no interaction withing slab, continue
        }
        t += dt;
        if (t >= d)
        {
            t = d + 0.00001;
            break; // reached the end of the segment
        }
        float density_val = extinction(_this, x + w * t);
        if (random() < density_val / maj) {
            break;
        }
    }
    _output[0] = t * density_scale; // free-flight distance in world space
}