
// cosine weighted sampler
FORWARD {
    float refraction_index = float_ptr(parameters.refraction_index).data[0];
    // incident direction
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    vec3 x = vec3(_input[3], _input[4], _input[5]);
    vec3 G = vec3(_input[6], _input[7], _input[8]);
    vec3 N = vec3(_input[9], _input[10], _input[11]);
    vec2 C = vec2(_input[12], _input[13]);
    bool from_outside = dot(win, G) < 0.0;
    vec3 fN = from_outside ? N : -N;
    vec3 reflect_w = reflect(win, fN);
    vec3 wout = win;
    float W = 1.0;
    float selected_lobe = random();
    if (selected_lobe < parameters.diffuse_prob)
    {
        // diffuse lobe
        wout = sample_cosine_hemisphere(fN);
        W = 1.0 / parameters.diffuse_prob;
    }
    else
    {
    }

    if (refraction_index > 1.001)
    {
        // surfel
        float eta = from_outside ? 1 / refraction_index : refraction_index;
        vec3 refract_w = refract(win, fN, eta);
        float theta = abs(dot(win, fN));
        float f = pow((1 - eta)/(1 + eta), 2);
        float R = f + (1 - f) * pow(1 - theta, 5);
        if (refract_w == vec3(0.0) || random() < R)
            wout = reflect_w;
        else
            wout = refract_w;
        W = 1.0/(eta*eta);
    }

    for (int i=0; i<SPECTRAL_DIM; i++)
    {
        _output[i] = 0.0;
        _output[i+SPECTRAL_DIM] = W;
    }

    x += wout * 0.0001;
    _output[2*SPECTRAL_DIM + 0] = x.x;
    _output[2*SPECTRAL_DIM + 1] = x.y;
    _output[2*SPECTRAL_DIM + 2] = x.z;
    _output[2*SPECTRAL_DIM + 3] = wout.x;
    _output[2*SPECTRAL_DIM + 4] = wout.y;
    _output[2*SPECTRAL_DIM + 5] = wout.z;

}