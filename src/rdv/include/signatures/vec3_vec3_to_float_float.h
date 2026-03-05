float SUBMAP_FORWARD_NAME(MAP_DECL, vec3 x, vec3 w, out float o) {
    float _output[2];
    forward(parameters.SUBMAP_NAME, float[6](x.x, x.y, x.z, w.x, w.y, w.z), _output);
    o = _output[1];
    return _output[0];
}

//#ifndef SUBMAP_BW_USES_OUTPUT
//#ifdef SUBMAP_INPUT_REQUIRES_GRAD
//void MAP_NAME_bw(MAP_DECL, vec3 x, vec3 w, float output_grad, inout vec3 x_grad, inout vec3 w_grad) {
//    float _input_grad[6];
//    backward(parameters.MAP_NAME, float[6](x.x, x.y, x.z, w.x, w.y, w.z), float[1](output_grad), _input_grad);
//    x_grad += vec3(_input_grad[0], _input_grad[1], _input_grad[2]);
//    w_grad += vec3(_input_grad[3], _input_grad[4], _input_grad[5]);
//}
//#else
//void MAP_NAME_bw(MAP_DECL, vec3 x, vec3 w, float output_grad) {
//    backward(parameters.MAP_NAME, float[6](x.x, x.y, x.z, w.x, w.y, w.z), float[1](output_grad));
//}
//#endif
//#else
//#ifdef SUBMAP_INPUT_REQUIRES_GRAD
//void MAP_NAME_bw(MAP_DECL, vec3 x, vec3 w, float output, float output_grad, inout vec3 x_grad, inout vec3 w_grad) {
//    float _input_grad[6];
//    backward(parameters.MAP_NAME, float[6](x.x, x.y, x.z, w.x, w.y, w.z), float[1](output), float[1](output_grad), _input_grad);
//    x_grad += vec3(_input_grad[0], _input_grad[1], _input_grad[2]);
//    w_grad += vec3(_input_grad[3], _input_grad[4], _input_grad[5]);
//}
//#else
//void MAP_NAME_bw(vec3 x, float output, float MAP_NAME_grad) {
//    backward(parameters.MAP_NAME, float[3](x.x, x.y, x.z), float[1](output), float[1](output_grad));
//}
//#endif
//#endif

#undef SUBMAP_NAME // make it here
#undef SUBMAP_BW_USES_OUTPUT
#undef SUBMAP_INPUT_REQUIRES_GRAD