FORWARD {
    float_ptr refraction_index_buf = float_ptr(parameters.refraction_indices);
    int sel_index = parameters.is_single==1 ? 0 : min(int(random()*SPECTRAL_DIM), SPECTRAL_DIM - 1);
    float refraction_index = refraction_index_buf.data[sel_index];
    // incident direction
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    vec3 x = vec3(_input[3], _input[4], _input[5]);
    vec3 wout = win;
    float W_rest = 1.0;
    float W_sel = 1.0;
    if (abs(refraction_index - 1.0) >= 0.0001)
    {
        // surfel
        vec3 G = vec3(_input[6], _input[7], _input[8]);
        vec3 N = vec3(_input[9], _input[10], _input[11]);
        //vec2 C = vec2(_input[12], _input[13]);
        bool from_outside = dot(win, G) < 0.0;
        vec3 fN = from_outside ? N : -N;
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
    for (int i=0; i<SPECTRAL_DIM; i++)
        _output[i] = i == sel_index ? W_sel : W_rest;
//        float P = event_R ? R : (1 - R);
//        W /= P;
//        for (int i=0; i<SPECTRAL_DIM; i++)
//        {
//            float ior_i = refraction_index_buf.data[i];
//            float eta_i = from_outside ? 1 / ior_i : ior_i;
//            float f_i = pow((1 - eta_i)/(1 + eta_i), 2);
//            float R_i = tir ? 1.0 : f_i + (1 - f_i) * pow(1 - theta, 5);
//            float F_i = event_R ? R_i : (1 - R_i);
//            _output[i] = W * F_i * ((i == sel_index ? SPECTRAL_DIM * 0.5 : 0.0) + 0.5);
////            _output[i] = W * F_i * (i == sel_index ? SPECTRAL_DIM : 0.0);
//        }
    _output[SPECTRAL_DIM + 0] = x.x;
    _output[SPECTRAL_DIM + 1] = x.y;
    _output[SPECTRAL_DIM + 2] = x.z;
    _output[SPECTRAL_DIM + 3] = wout.x;
    _output[SPECTRAL_DIM + 4] = wout.y;
    _output[SPECTRAL_DIM + 5] = wout.z;

}