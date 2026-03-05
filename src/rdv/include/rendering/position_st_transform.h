/*
input: vec3 x
output: vec3 x' = scale * x + offset
parameters:
- scale: vec3
- offset: vec3
*/

FORWARD {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    x *= parameters.scale;
    x += parameters.offset;
    _output = float[3](x.x, x.y, x.z);
}

BACKWARD {
#ifdef INPUT_REQUIRES_GRAD
    vec3 x_grad = vec3(_output_grad[0], _output_grad[1], _output_grad[2]);
    vec3 input_grad = x_grad * parameters.scale;
    _input_grad = float[3](input_grad.x, input_grad.y, input_grad.z);
#endif
}