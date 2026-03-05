#define SUBMAP_NAME extinction
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME majorant
#include "signatures/vec3_vec3_to_float_float.h"

#define SUBMAP_NAME anisotropy
#include "signatures/vec3_to_float.h"

#define SUBMAP_NAME scattering_albedo
#include "signatures/vec3_to_spectral.h"

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];

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
            break; // reached the end of the segment

        float density_val = extinction(_this, x + w * t);
        if (random() < density_val / maj) {
            break;
        }
    }

    x = vec3(_input[0], _input[1], _input[2]); // initial position in world space
    w = vec3(_input[3], _input[4], _input[5]); // initial direction in world space
    x += w * min(t, d) * density_scale; // transform point to world space
    for (int i=0; i<SPECTRAL_DIM; i++)
        _output[i] = 0.0;

    if (t < d) // integrate interation within medium
    {
        float sa[SPECTRAL_DIM];
        scattering_albedo(_this, x, sa);
        for (int i=0; i<SPECTRAL_DIM; i++)
            _output[i + SPECTRAL_DIM] = sa[i];
        float g = anisotropy(_this, x);
        float pdf;
        w = hg_phase_sample(w, g, pdf);
    }
    else {
        for (int i=0; i<SPECTRAL_DIM; i++)
            _output[i + SPECTRAL_DIM] = 1.0;
    }
    _output[SPECTRAL_DIM*2 + 0] = x.x; // xout
    _output[SPECTRAL_DIM*2 + 1] = x.y;
    _output[SPECTRAL_DIM*2 + 2] = x.z;
    _output[SPECTRAL_DIM*2 + 3] = w.x; // wout
    _output[SPECTRAL_DIM*2 + 4] = w.y;
    _output[SPECTRAL_DIM*2 + 5] = w.z;
}

BACKWARD {
}