/*
Unproject equirectangular coordinates (u, v) to 3D points on the unit sphere.
*/

FORWARD {
    vec3 w = xr2dir(vec2(_input[0], _input[1]));
    _output = float[3](w.x, w.y, w.z);
}

BACKWARD {
    NOT_SUPPORTED("[ERROR] Backward equirectangular_inverse_projection is not supported.");
}