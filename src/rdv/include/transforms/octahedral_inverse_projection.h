/*
Unproject a 2D point from octahedral map to a 3D point on the unit sphere.
*/

FORWARD {
    vec3 w = oct2dir(vec2(_input[0], _input[1]));
    _output = float[3](w.x, w.y, w.z);
}

BACKWARD {
    NOT_SUPPORTED("[ERROR] Backward octahedral_inverse_projection is not supported.");
}