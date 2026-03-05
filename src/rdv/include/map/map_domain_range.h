/*
input: vector
output: vector
parameters:
- map: function to be masked
- range_min
- range_max
*/

FORWARD {
    float_ptr min_buf = float_ptr(parameters.range_min);
    float_ptr max_buf = float_ptr(parameters.range_max);
    for (int i=0; i<INPUT_DIM; i++)
        if (_input[i] < min_buf.data[0] || _input[i] > max_buf.data[0])
        {
            for (int j=0; j<OUTPUT_DIM; j++)
                _output[j] = 0.0;
           return;
        }
    forward(parameters.map, _input, _output);
}

BACKWARD {
    float_ptr min_buf = float_ptr(parameters.range_min);
    float_ptr max_buf = float_ptr(parameters.range_max);
    for (int i=0; i<INPUT_DIM; i++)
        if (_input[i] < min_buf.data[0] || _input[i] > max_buf.data[0])
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