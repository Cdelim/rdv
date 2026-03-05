/*
input: x, w (ray origin and direction)
output: x', w' (transformed ray origin and direction)
*/

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    x = parameters.scale * x + parameters.offset;
    w = normalize(parameters.scale * w);
    _output = float[6]( x.x, x.y, x.z, w.x, w.y, w.z );
}

BACKWARD {
#ifdef INPUT_REQUIRES_GRAD
    vec3 x_grad = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    vec3 w_grad = vec3(_output_grad[3], _output_grad[4], _output_grad[5]);
    vec3 input_x_grad = x_grad * parameters.scale;
    vec3 input_w_grad = normalize(parameters.scale) * (w_grad - dot(w, w_grad) * w);
    _input_grad = float[6](input_x_grad.x, input_x_grad.y, input_x_grad.z,
                          input_w_grad.x, input_w_grad.y, input_w_grad.z);
#endif
}