float SUBMAP_FORWARD_NAME(MAP_DECL, vec3 x, vec3 w, float a, float b) {
    float _output[1];
    forward(parameters.SUBMAP_NAME, float[](x.x, x.y, x.z, w.x, w.y, w.z, a, b), _output);
    return _output[0];
}

#ifndef SUBMAP_BW_USES_OUTPUT
#ifdef SUBMAP_INPUT_REQUIRES_GRAD
void SUBMAP_BACKWARD_NAME(MAP_DECL, vec3 x, vec3 w, float a, float b, float output_grad, inout vec3 x_grad, inout vec3 w_grad, inout float a_grad, inout float b_grad) {
    float _input_grad[8];
    backward(parameters.SUBMAP_NAME, float[](x.x, x.y, x.z, w.x, w.y, w.z, a, b), float[1](output_grad), _input_grad);
    x_grad += vec3(_input_grad[0], _input_grad[1], _input_grad[2]);
    w_grad += vec3(_input_grad[3], _input_grad[4], _input_grad[5]);
    a_grad += _input_grad[6];
    b_grad += _input_grad[7];
}
#else
void SUBMAP_BACKWARD_NAME(MAP_DECL, vec3 x, vec3 w, float a, float b, float output_grad) {
    backward(parameters.SUBMAP_NAME, float[](x.x, x.y, x.z, w.x, w.y, w.z, a, b), float[1](output_grad));
}
#endif
#else
#ifdef SUBMAP_INPUT_REQUIRES_GRAD
void SUBMAP_BACKWARD_NAME(MAP_DECL, vec3 x, vec3 w, float a, float b, float o, float o_grad, inout vec3 x_grad, inout vec3 w_grad, inout float a_grad, inout float b_grad) {
    float _input_grad[8];
    backward(parameters.SUBMAP_NAME, float[](x.x, x.y, x.z, w.x, w.y, w.z, a, b), float[](o), float[](o_grad), _input_grad);
    x_grad += vec3(_input_grad[0], _input_grad[1], _input_grad[2]);
    w_grad += vec3(_input_grad[3], _input_grad[4], _input_grad[5]);
    a_grad += _input_grad[6];
    b_grad += _input_grad[7];
}
#else
void SUBMAP_BACKWARD_NAME(MAP_DECL, vec3 x, vec3 w, float a, float b, float o, float o_grad) {
    backward(parameters.SUBMAP_NAME, float[](x.x, x.y, x.z, w.x, w.y, w.z, a, b), float[](o), float[](o_grad));
}
#endif
#endif

#undef SUBMAP_NAME // make it here
#undef SUBMAP_BW_USES_OUTPUT
#undef SUBMAP_INPUT_REQUIRES_GRAD