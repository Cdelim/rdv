FORWARD {
    // incident direction
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    vec3 x = vec3(_input[3], _input[4], _input[5]);
    vec3 G = vec3(_input[6], _input[7], _input[8]);
    bool from_outside = dot(win, G) < 0.0;
    vec3 N = vec3(_input[9], _input[10], _input[11]);
    vec3 fN = from_outside ? N : -N;
    // x += fN * 0.001;
    _output[SPECTRAL_DIM + 0] = x.x;
    _output[SPECTRAL_DIM + 1] = x.y;
    _output[SPECTRAL_DIM + 2] = x.z;
    float NdotD;
    vec3 wout = random_direction_HS_cosine_weighted(fN, NdotD);
    _output[SPECTRAL_DIM + 3] = wout.x;
    _output[SPECTRAL_DIM + 4] = wout.y;
    _output[SPECTRAL_DIM + 5] = wout.z;
    float albedo[SPECTRAL_DIM];
    forward(parameters.albedo, float[](_input[12] * 2 - 1, _input[13] * 2 - 1), albedo);
    for (int i=0; i<SPECTRAL_DIM; i++)
        _output[i] = albedo[i];
}