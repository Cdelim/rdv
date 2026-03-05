/*
Maps sphere coordinates to octahedral 2D coordinates in GLSL
*/
FORWARD {
    vec2 x = dir2oct(vec3(_input[0], _input[1], _input[2]));
    _output = float[2](x.x, x.y);
}

BACKWARD {
    NOT_SUPPORTED("[ERROR] Backward octahedral_sphere_projection is not supported.");
}