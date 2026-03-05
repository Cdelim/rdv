FORWARD {
    // input ray
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float scattering_albedo[SPECTRAL_DIM];
    forward(parameters.scattering_albedo, float[](x.x, x.y, x.z), scattering_albedo);
    float phase_sampler[5]; //wo weight(wo) 1/pdf(wo)
    forward(parameters.phase_function_sampler, float[](x.x, x.y, x.z, w.x, w.y, w.z), phase_sampler);
    for (int i = 0; i<SPECTRAL_DIM; i++)
        _output[i] = scattering_albedo[i] * phase_sampler[3]; // throughput initialized with scattering albedo
    _output[SPECTRAL_DIM + 0] = x.x;
    _output[SPECTRAL_DIM + 1] = x.y;
    _output[SPECTRAL_DIM + 2] = x.z;
    _output[SPECTRAL_DIM + 3] = phase_sampler[0]; // scattered direction x
    _output[SPECTRAL_DIM + 4] = phase_sampler[1]; // scattered direction y
    _output[SPECTRAL_DIM + 5] = phase_sampler[2]; // scattered direction z
}