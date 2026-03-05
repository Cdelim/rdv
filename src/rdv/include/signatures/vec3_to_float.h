float SUBMAP_FORWARD_NAME(MAP_DECL, vec3 x) {
    float _output[1];
    forward(parameters.SUBMAP_NAME, float[3](x.x, x.y, x.z), _output);
    return _output[0];
}

#ifndef SUBMAP_BW_USES_OUTPUT
#ifdef SUBMAP_INPUT_REQUIRES_GRAD
void SUBMAP_BACKWARD_NAME(MAP_DECL, vec3 x, float output_grad, inout vec3 x_grad) {
    float _x_grad[3];
    backward(parameters.SUBMAP_NAME, float[3](x.x, x.y, x.z), float[1](output_grad), _x_grad);
    x_grad += vec3(_x_grad[0], _x_grad[1], _x_grad[2]);
}
#else
void SUBMAP_BACKWARD_NAME(MAP_DECL, vec3 x, float output_grad) {
    backward(parameters.SUBMAP_NAME, float[3](x.x, x.y, x.z), float[1](output_grad));
}
#endif
#else
#ifdef SUBMAP_INPUT_REQUIRES_GRAD
void SUBMAP_BACKWARD_NAME(MAP_DECL, vec3 x, float o, float o_grad, inout vec3 x_grad) {
    float _x_grad[3];
    backward(parameters.SUBMAP_NAME, float[3](x.x, x.y, x.z), float[1](o), float[1](o_grad), _x_grad);
    x_grad += vec3(_x_grad[0], _x_grad[1], _x_grad[2]);
}
#else
void SUBMAP_BACKWARD_NAME(MAP_DECL, vec3 x, float o, float o_grad) {
    backward(parameters.SUBMAP_NAME, float[3](x.x, x.y, x.z), float[1](o), float[1](o_grad));
}
#endif
#endif

#undef SUBMAP_NAME // make it here
#undef SUBMAP_BW_USES_OUTPUT
#undef SUBMAP_INPUT_REQUIRES_GRAD
