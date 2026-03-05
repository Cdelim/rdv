// input: x, w, max_t
// output: A, W, xout, wout

// event_sampler: x, w, max_t -> t, event_info
// scattering_sampler: xt, w, event_info -> W, wout, pdf(wout)

FORWARD {
    // Sample scattering event distance
    float event_sampler_output[1 + EVENT_INFO_DIM];
    forward(parameters.event_sampler, _input, event_sampler_output);

    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    x += w * event_sampler_output[0]; // x at scattering event

    if (event_sampler_output[0] < _input[6]) // interaction occurs before max_t
    {
        float scattering_sampler_output[3 + SPECTRAL_DIM + 1];
        float scattering_sampler_input[6 + EVENT_INFO_DIM];
        scattering_sampler_input[0] = _input[0] + _input[3] * event_sampler_output[0]; // x
        scattering_sampler_input[1] = _input[1] + _input[4] * event_sampler_output[0];
        scattering_sampler_input[2] = _input[2] + _input[5] * event_sampler_output[0];
        scattering_sampler_input[3] = _input[3]; // w
        scattering_sampler_input[4] = _input[4];
        scattering_sampler_input[5] = _input[5];
        for(int i=0; i < EVENT_INFO_DIM; i++)
            scattering_sampler_input[6 + i] = event_sampler_output[1 + i];
        // Sample scattering event properties
        forward(parameters.scattering_sampler, scattering_sampler_input, scattering_sampler_output);
        for (int i=0; i<SPECTRAL_DIM; i++)
            _output[i] = 0.0; // TODO: add emission here
        for (int i=0; i<SPECTRAL_DIM; i++)
            _output[i + SPECTRAL_DIM] = scattering_sampler_output[3 + i]; // path-throughput in scattering event
        w = vec3(scattering_sampler_output[0], scattering_sampler_output[1], scattering_sampler_output[2]);
    }
    else {
        for (int i=0; i<SPECTRAL_DIM; i++)
            _output[i] = 0.0;
        for (int i=0; i<SPECTRAL_DIM; i++)
            _output[i + SPECTRAL_DIM] = 1.0;
    }
    _output[SPECTRAL_DIM*2 + 0] = x.x;
    _output[SPECTRAL_DIM*2 + 1] = x.y;
    _output[SPECTRAL_DIM*2 + 2] = x.z;
    _output[SPECTRAL_DIM*2 + 3] = w.x;
    _output[SPECTRAL_DIM*2 + 4] = w.y;
    _output[SPECTRAL_DIM*2 + 5] = w.z;
}