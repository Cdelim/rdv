/*
parameters:
- base_transmittance: ray segment + sigma_scale -> float transmittance
- scale, offset: vec3
  Defines the scaling factor and offset applied to the space before querying the base_transmittance.
*/

void transform_input(MAP_DECL, float _input[8], out float _new_input[8]) {
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);
    x *= parameters.scale;
    x += parameters.offset;
    w *= parameters.scale;
    float s = length(w);
    w /= s; // normalize w
    _new_input = float[](x.x, x.y, x.z, w.x, w.y, w.z, _input[6] * s, _input[7] / s);
}

FORWARD {
    float _new_input[8];
    transform_input(_this, _input, _new_input);
    forward (parameters.base_transmittance, _new_input, _output);
}

BACKWARD {
    float _new_input[8];
    transform_input(_this, _input, _new_input);
#ifdef INPUT_REQUIRES_GRAD
    #ifndef BW_USES_OUTPUT
    backward(parameters.base_transmittance, _new_input, _output_grad, _input_grad);
    #else
    backward(parameters.base_transmittance, _new_input, _output, _output_grad, _input_grad);
    #endif
#else
    #ifndef BW_USES_OUTPUT
    backward(parameters.base_transmittance, _new_input, _output_grad);
    #else
    backward(parameters.base_transmittance, _new_input, _output, _output_grad);
    #endif
#endif
}