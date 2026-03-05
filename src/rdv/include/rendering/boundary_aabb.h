FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float tmin, tmax;
    ray_box_intersection(x, w, parameters.bmin, parameters.bmax, tmin, tmax);
    if (tmin >= tmax || tmax <= 0.0) {
        _output[0] = POSINF; // no intersection
    } else {
        if (tmin <= 0.0) {
            _output[0] = -tmax; // inside box, distance is given negative
        } else {
            _output[0] = tmin; // outside box
        }
    }
}

BACKWARD {
    NOT_SUPPORTED("Boundary AABB backward pass is not supported.");
}