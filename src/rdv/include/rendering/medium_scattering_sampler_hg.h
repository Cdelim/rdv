#define SUBMAP_NAME anisotropy
#include "signatures/vec3_to_float.h"

FORWARD
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float g = anisotropy(_this, x);
    float pdf;
    vec3 wout = hg_phase_sample(w, g, pdf);
    _output[0] = wout.x;
    _output[1] = wout.y;
    _output[2] = wout.z;
    _output[SPECTRAL_DIM + 3] = pdf;
    float sa [SPECTRAL_DIM];
    forward(parameters.scattering_albedo, float[](x.x, x.y, x.z), sa);
    for(int i=0; i < SPECTRAL_DIM; i++)
        _output[3 + i] = sa[i];
}