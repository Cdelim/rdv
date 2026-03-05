#define SUBMAP_NAME anisotropy
#include "signatures/vec3_to_float.h"

FORWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 wout = vec3(_input[6], _input[7], _input[8]);
    float g = anisotropy(_this, x);
    float hg = hg_phase_eval(w, wout, g);
    forward(parameters.scattering_albedo, float[](x.x, x.y, x.z), _output);
    for(int i=0; i < OUTPUT_DIM; i++)
        _output[i] *= hg;
}