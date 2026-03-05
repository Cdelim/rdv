FORWARD {
    // incident direction
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    // outgoing direction
    vec3 wout = vec3(_input[3], _input[4], _input[5]);
    // surface coordinate
    vec2 uv = vec2(_input[6 + UV_INDEX], _input[6 + UV_INDEX + 1]);
    if (win.z < 0.0 || wout.z < 0.0) { // not in the upper hemisphere
        for (int i=0; i<SPECTRAL_DIM; i++)
            _output[i] = 0.0;
        return;
    }
    forward(parameters.albedo, float[](uv.x * 2 - 1, uv.y * 2 - 1), _output);
    float bsdf = wout.z / pi; // 1/pi times cosine term
    for (int i=0; i<SPECTRAL_DIM; i++)
        _output[i] *= bsdf;
}