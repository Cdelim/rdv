/*
parameters:
- phase_g: map vec3 -> float
*/

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float g[1];
    forward(parameters.g_factor, float[](x.x, x.y, x.z), g);
    float phase_pdf;
    vec3 wo = hg_phase_sample(w, g[0], phase_pdf);
    _output[0] = wo.x;
    _output[1] = wo.y;
    _output[2] = wo.z;
    _output[3] = 1.0; // perfect importance sampled
    _output[4] = 1/phase_pdf;
}