/*
Equirectangular Sphere Projection Transform. From sphere 3D point to 2D point on equirectangular map.
*/

FORWARD {
    vec2 x = dir2xr(vec3(_input[0], _input[1], _input[2]));
    _output = float[2](x.x, x.y);
}

BACKWARD {
    NOT_SUPPORTED("[ERROR] Backward equirectangular_sphere_projection is not supported.");
}

