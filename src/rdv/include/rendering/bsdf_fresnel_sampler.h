FORWARD {
    float_ptr refraction_index_buf = float_ptr(parameters.refraction_indices);
    int sel_index = parameters.is_single == 1 ? 0 : min(int(random()*SPECTRAL_DIM), SPECTRAL_DIM - 1);
    float refraction_index = refraction_index_buf.data[sel_index];

    // incident direction
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    vec3 wout = win;
    float W_rest = 1.0;
    float W_sel = 1.0;
    if (abs(refraction_index - 1.0) >= 0.0001)
    {
        bool from_outside = win.z < 0.0;
        vec3 fN = vec3(0, 0, from_outside ? 1 : -1);
        float eta = from_outside ? 1 / refraction_index : refraction_index;
        vec3 refract_w = refract(win, fN, eta);
        float f = pow((1 - eta)/(1 + eta), 2);
        bool tir = all(equal(refract_w, vec3(0.0)));
        float theta = abs(dot(win, fN));
        float R = tir ? 1 : (f + (1 - f) * pow(1 - theta, 5));
        bool event_R = random() < R;
        wout = event_R ? reflect(win, fN) : refract_w;
        float W = event_R ? 1.0 : 1.0 / (eta * eta); // account for solid angle change with reference IOR
        if (parameters.is_single==1)
        {
            W_sel = W;
            W_rest = W;
        }
        else
        {
            float amount = 0.66;
            W_sel = W * (SPECTRAL_DIM * amount + 1 - amount);
            W_rest = W * (1 - amount);
        }
    }
    _output[0] = wout.x;
    _output[1] = wout.y;
    _output[2] = wout.z;
    for (int i=0; i<SPECTRAL_DIM; i++)
        _output[3 + i] = i == sel_index ? W_sel : W_rest;
    _output[SPECTRAL_DIM + 3] = POSINF; // delta distribution indicator
}