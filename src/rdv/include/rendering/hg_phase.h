/*
parameters:
- g_factor: map vec3 -> float
*/
FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    vec3 wo = vec3(_input[6], _input[7], _input[8]);
    float g[1];
    forward(parameters.g_factor, float[](x.x, x.y, x.z), g);
    float rho = hg_phase_eval(w, wo, g[0]);
    _output[0] = rho;
}

BACKWARD {
    NOT_IMPLEMENTED("hg_phase backward not implemented");
}