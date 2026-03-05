/*
scale: scaling factor
offset: offset to be added after scaling
*/

FORWARD {
    float_ptr scale = float_ptr(parameters.scale);
    float_ptr offset = float_ptr(parameters.offset);
    for (int i=0; i<OUTPUT_DIM; i++)
        _output[i] = _input[i] * scale.data[i] + offset.data[i];
}

BACKWARD {
#ifdef INPUT_REQUIRES_GRAD
    float_ptr scale = float_ptr(parameters.scale);
    for (int i=0; i<OUTPUT_DIM; i++)
        _input_grad[i] += _output_grad[i] * scale.data[i];
#endif
}