FORWARD {
    // incident direction
    vec3 win = vec3(_input[0], _input[1], _input[2]);
    // outgoing direction
    vec3 wout = vec3(_input[3], _input[4], _input[5]);
    // pdf eval of scattering from win to wout
    _output[0] = win.z < 0.0 || wout.z < 0.0 ? 0.0 : wout.z / pi;
}