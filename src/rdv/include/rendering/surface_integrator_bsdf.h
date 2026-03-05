FORWARD {
    float bsdf_sampler_input[3 + SURFEL_DIM];
    // win
    vec3 win = vec3(_input[3], _input[4], _input[5]);
    // get normal
    vec3 N = vec3(_input[6 + SHADING_NORMAL_INDEX], _input[6 + SHADING_NORMAL_INDEX + 1], _input[6 + SHADING_NORMAL_INDEX + 2]);
    // get tangent
    vec3 T = vec3(_input[6 + TANGENT_INDEX], _input[6 + TANGENT_INDEX + 1], _input[6 + TANGENT_INDEX + 2]);
    vec3 B = cross(N, T);
    // transform win to local space
    bsdf_sampler_input[0] = dot(win, T);
    bsdf_sampler_input[1] = dot(win, B);
    bsdf_sampler_input[2] = dot(win, N);
    // surfel
    for (int i = 0; i < SURFEL_DIM; i++)
        bsdf_sampler_input[3 + i] = _input[6 + i];
    float bsdf_sampler_output[3 + SPECTRAL_DIM + 1]; // wout | weight | pdf
    forward(parameters.bsdf_sampler, bsdf_sampler_input, bsdf_sampler_output);
    // TODO: Collect statistics about sampled bsdf in path state
    for (int i=0; i<PATH_STATE_DIM; i++) { // copy path state unaltered
        _output[2*SPECTRAL_DIM + 6 + i] = _input[6 + SURFEL_DIM + i];
    }
    // xout:  bsdfs doesnt change x
    _output[SPECTRAL_DIM*2 + 0] = _input[0];
    _output[SPECTRAL_DIM*2 + 1] = _input[1];
    _output[SPECTRAL_DIM*2 + 2] = _input[2];
    // wout
    vec3 wout = bsdf_sampler_output[0] * T + bsdf_sampler_output[1] * B + bsdf_sampler_output[2] * N;
    _output[SPECTRAL_DIM*2 + 3] = wout.x;
    _output[SPECTRAL_DIM*2 + 4] = wout.y;
    _output[SPECTRAL_DIM*2 + 5] = wout.z;
    // W
    for (int i = 0; i < SPECTRAL_DIM; i++)
        _output[SPECTRAL_DIM + i] = bsdf_sampler_output[3 + i];
    // emission
    float emission[SPECTRAL_DIM];
    forward(parameters.emission, bsdf_sampler_input, emission);
    for (int i = 0; i < SPECTRAL_DIM; i++)
        _output[i] = emission[i];
}