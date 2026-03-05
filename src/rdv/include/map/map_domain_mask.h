/*
input: vector
output: vector
parameters:
- map: function to be masked
- mask: boolean map mask
*/

FORWARD {
    float _output_mask[1];
    forward(parameters.mask, _input, _output_mask);
    if (_output_mask[0] <= 0.0)
    {
        for (int j=0; j<OUTPUT_DIM; j++)
            _output[j] = 0.0;
        return;
    }
    forward(parameters.map, _input, _output);
}

BACKWARD {
    float _output_mask[1];
    forward(parameters.mask, _input, _output_mask);
    if (_output_mask[0] <= 0.0)
        return;
#ifdef INPUT_REQUIRES_GRAD
    #ifndef BW_USES_OUTPUT
    backward(parameters.map, _input, _output_grad, _input_grad);
    #else
    backward(parameters.map, _input, _output, _output_grad, _input_grad);
    #endif
#else
    #ifndef BW_USES_OUTPUT
    backward(parameters.map, _input, _output_grad);
    #else
    backward(parameters.map, _input, _output, _output_grad);
    #endif
#endif
}