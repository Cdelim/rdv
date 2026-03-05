/*
parameters:
transform: mat4x3
Takes a ray, transforms it into the local space defined by the inverse of the given transform.
intersect with unitary box [-1,1]^3, then transform back the hit to world space.
*/

FORWARD {
    Tensor T = load_deferred(parameters.transform);
    mat4x3 M = T.data_ptr != 0 ? mat4x3_ptr(T.data_ptr).data[0] : mat4x3(1.0);
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    transform_ray_to_object(x, w, M);
    float tMin, tMax;
    ray_box_intersection(x, w, tMin, tMax);
    bool is_inside = tMax > 0.0 && tMin < 0.0;
    float t = is_inside ? tMax : tMax < 0 || tMin >= tMax ? POSINF : tMin;
    x += w * t; // locate in box
    _output = float[](
        intBitsToFloat(t == POSINF ? -1: 0),
        t * (is_inside ? -1.0 : 1.0), // hit_t with sign
        x.x, x.y, x.z // hit point in object space to recover surfel UVs later
    );
}