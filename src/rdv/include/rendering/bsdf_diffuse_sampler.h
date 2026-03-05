FORWARD {
    // incident direction
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    // surface coordinate
    vec2 uv = vec2(_input[3 + UV_INDEX], _input[3 + UV_INDEX + 1]);
    if (win.z > 0.0) { // not in the upper hemisphere
        for (int i=0; i<3 + SPECTRAL_DIM + 1; i++)
            _output[i] = 0.0; // wout, weight(wout)
        return;
    }
    vec3 wout = random_direction_HS_cosine_weighted();
    _output[0] = wout.x;
    _output[1] = wout.y;
    _output[2] = wout.z;
    float _albedo[SPECTRAL_DIM];
    forward(parameters.albedo, float[](uv.x * 2 - 1, uv.y * 2 - 1), _albedo);
    for (int i=0; i<SPECTRAL_DIM; i++)
        _output[3 + i] = _albedo[i];
    _output[3 + SPECTRAL_DIM] = win.z > 0.0 || wout.z < 0.0 ? 0.0 : wout.z / pi; // pdf is cos(theta)/pi, weight is bsdf/pdf = albedo * (cos(theta)/pi) / (cos(theta)/pi) = albedo
}