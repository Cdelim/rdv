/*
input: x, w, d, distance_scale (ray origin, direction, distance, scale)
output: x', w', d', distance_scale' (transformed ray origin, direction, distance, scale)
parameters:
-scale: vec3
-offset: vec3
*/

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    float d = _input[6];
    float distance_scale = _input[7];
    x = parameters.scale * x + parameters.offset;
    w = parameters.scale * w;
    float s = length(w);
    w /= s; // normalize w
    d *= s;
    distance_scale /= s;
    _output = float[8]( x.x, x.y, x.z, w.x, w.y, w.z, d, distance_scale );
}

BACKWARD {
#ifdef INPUT_REQUIRES_GRAD
    vec3 x_grad = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    vec3 w_grad = vec3(_output_grad[3], _output_grad[4], _output_grad[5]);
    float d_grad = _output_grad[6];
    float distance_scale_grad = _output_grad[7];

    vec3 input_x_grad = x_grad * parameters.scale;
    vec3 input_w_grad = normalize(parameters.scale) * (w_grad - dot(w, w_grad) * w);
    float s = length(parameters.scale * w);
    float input_d_grad = d_grad * s;
    float input_distance_scale_grad = distance_scale_grad / s;

    _input_grad = float[8](input_x_grad.x, input_x_grad.y, input_x_grad.z,
                          input_w_grad.x, input_w_grad.y, input_w_grad.z,
                          input_d_grad,
                          input_distance_scale_grad);
#endif
}
