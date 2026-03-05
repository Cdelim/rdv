// Given a direction w, it returns the SH coefficients

FORWARD {
    vec3 w = vec3(_input[0], _input[1], _input[2]);
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = 0.0;
    float sh_coeffs[OUTPUT_DIM];
    sh_project(w, sh_coeffs);
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = sh_coeffs[i];
}